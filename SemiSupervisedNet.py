'''
@author: Dmitry
'''

import tensorflow as tf

from Trainer import *
from Runners import *

class ConvData():
    def __init__(self, num_features, kernel, stride, padding = 'same', activation = None):
        self.num_features, self.kernel, self.stride = num_features, kernel, stride
        self.padding = padding
        self.activation = activation

    def get_conv_data(self):
        return self.num_features, self.kernel, self.stride, self.padding

class SemiSupervisedConfig():
    def __init__(self, num_outputs = 10, num_d_conv = 3, num_g_conv = 3, alpha = 0.01, start_features = 16, conv_kernel = 3, conv_stride = 2,
                 code_size = 80, first_decode_shape = [4, 4, 16], orig_channels = 3, mu = 0.1):
        self.alpha = alpha
        self.d_convs = []
        self.g_convs = []

        n_features = start_features
        for i in range(num_d_conv):
            self.d_convs.insert(0, ConvData(n_features, conv_kernel, conv_stride))
            n_features *= 2

        n_features = start_features
        for i in range(num_g_conv):
            self.g_convs.append(ConvData(n_features, conv_kernel, conv_stride))
            n_features *= 2

        self.code_size = code_size
        self.num_outputs = num_outputs
        self.first_decode_shape = first_decode_shape
        self.orig_channels = orig_channels # derive from input?
        self.mu = mu

class SemiSupervisedNetwork():
    def __init__(self, config = SemiSupervisedConfig()):
        self.config = config

    def get_code(self, images, reuse = False, training = True):
        config = self.config

        cur = images
        with(tf.variable_scope('discriminator', reuse=reuse)):
            for i, conv in enumerate(config.d_convs):
                cur = tf.layers.conv2d(cur, conv.num_features, conv.kernel, conv.stride, padding = 'same', name = 'dconv%d' % i)
                cur = tf.layers.batch_normalization(cur, training = training, name = 'dbatch%d' % i)
                cur = leaky_relu(cur, config.alpha)
            cur = tf.layers.flatten(cur)
            labels, code = tf.layers.dense(cur, config.num_outputs), tf.layers.dense(cur, config.code_size, activation = tf.tanh)
        return labels, code

    def get_decoded(self, encoded, reuse = False, training = True):
        config = self.config

        with(tf.variable_scope('generator', reuse=reuse)):
            dense_size = np.prod(config.first_decode_shape)
            cur = tf.layers.dense(encoded, dense_size, activation = tf.tanh)
            cur = tf.reshape(cur, [-1, *config.first_decode_shape])
            w, h = config.first_decode_shape[1], config.first_decode_shape[0]

            for conv in config.g_convs:
                cur = tf.layers.conv2d_transpose(cur, conv.num_features, conv.kernel, conv.stride, padding = 'same')
                cur = tf.layers.batch_normalization(cur, training = training)
                cur = leaky_relu(cur, config.alpha)
                w, h = 2 * w, 2 * h
                #cur = tf.image.resize_images(cur, (w, h), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            cur = tf.layers.conv2d(cur, config.orig_channels, 1, padding = 'same')
        return cur

    def run_autoencoder(self, inputs, training = True):
        self.ae_inputs = inputs
        self.labels, self.code = self.get_code(inputs, training = training)
        encoded = tf.concat((tf.nn.softmax(self.labels), tf.tanh(self.code)), axis = 1)
        self.decoded = self.get_decoded(encoded, training = training)
        self.img = tf.tanh(self.decoded)

    def ae_loss(self, targets, images):
        ae_loss = tf.reduce_sum(tanh_cross_entropy(self.decoded, images))
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classification_loss = tf.reduce_sum(labels_mask * tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = self.labels))
        return classification_loss + self.config.mu * ae_loss

    def get_num_classified(self, targets, labels):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classified = tf.equal(tf.argmax(labels, axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)

    def num_classified_ae(self, targets, images):
        return self.get_num_classified(targets, self.labels)

    def get_minimizers_ae(self, optimizer, loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizers = [optimizer.minimize(l) for l in loss]
        return minimizers

    def is_fake(self, code, reuse = False):
        with (tf.variable_scope('discriminator_add', reuse = reuse)):
            fake_detector = tf.layers.dense(code, 1)
        return fake_detector

    def run_semi_supervised(self, images, randoms, training = True):
        self.generated = tf.tanh(self.get_decoded(randoms, reuse = tf.AUTO_REUSE, training = training))
        self.gen_labels = randoms[:, :self.config.num_outputs]

        classification, codes = self.get_code(tf.concat((images, self.generated), axis = 0), reuse = tf.AUTO_REUSE, training = training)
        sizes = [tf.shape(images)[0], tf.shape(randoms)[0]]
        self.classification_orig, self.classification_fake = tf.split(classification, sizes)
        #code_orig, code_fake = tf.split(codes, sizes)
        self.orig, self.fakes = tf.split(self.is_fake(tf.tanh(codes)), sizes)
        #self.fakes = self.is_fake(tf.tanh(code_fake), reuse = True)

    def semi_loss(self, targets, images):
        labels_mask = tf.reduce_sum(targets, axis = -1)

        classification_loss_d = tf.reduce_sum(labels_mask * tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = self.classification_orig))
        classification_loss_g = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.gen_labels, logits = self.classification_fake))
        d_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.orig), logits = self.orig))
        d_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.fakes), logits = self.fakes))
        d_loss += classification_loss_d # + 0.1 * classification_loss_g

        g_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.fakes), logits = self.fakes)) # + 0.1 * classification_loss_g
        #g_loss = classification_loss_g
        return d_loss, g_loss

    def num_classified_semi(self, targets, images):
        return self.get_num_classified(targets, self.classification_orig)

    def get_minimizers_semi(self, optimizer, loss):
        all_vars = tf.trainable_variables()
        d_vars = [v for v in all_vars if v.name.startswith('discriminator')]
        g_vars = [v for v in all_vars if v.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizer_d = optimizer.minimize(loss[0], var_list = d_vars)
            minimizer_g = optimizer.minimize(loss[1], var_list = g_vars)
            #minimizer = optimizer.optimize(loss[0], loss[1], d_vars, g_vars)
        return [minimizer_d, minimizer_g]

    def get_ae_functions_for_trainer(self):
        return 'ae_loss', 'run_autoencoder', 'num_classified_ae', 'get_minimizers_ae'

    def get_semi_functions_for_trainer(self):
        return 'semi_loss', 'run_semi_supervised', 'num_classified_semi', 'get_minimizers_semi'












class BasicNet():
    def __init__(self, modes_dict, mode, elements_dict = None, config = None):
        self.modes = modes_dict
        self.config = config
        self.runner = None

        self.mode = None
        self.set_mode(mode)
        self.run_elements = None
        self.set_elements(elements_dict)

    def set_mode(self, mode):
        self.mode = mode
        self.runner = self.modes[self.mode]

    def set_elements(self, elements_dict):
        self.run_elements = elements_dict

    def run(self, *args, **kwargs):
        print('Run:', self.mode)
        return self.runner.run(self.config, self.run_elements, *args, **kwargs)

    def loss_function(self, *args, **kwargs):
        print('Loss:', self.mode)
        return self.runner.loss_function(self.config, *args, **kwargs)

    def num_classified(self, *args, **kwargs):
        return self.runner.num_classified(self.config, *args, **kwargs)

    def get_minimizers(self, *args, **kwargs):
        return self.runner.get_minimizers(*args, **kwargs)

    def get_functions_for_trainer(self):
        return 'loss_function', 'run', 'num_classified', 'get_minimizers'

    def get_runner(self, mode = None):
        return self.runner if mode is None else self.modes[mode]

class SemiCapsCodeGenerator():
    def __init__(self, num_outputs, caps_len):
        self.num_outputs = num_outputs
        self.caps_len = caps_len
        self.code_size = num_outputs * caps_len

    def get_code(self, batch_size):
        random_targets = np.eye(self.num_outputs)[np.random.randint(0, self.num_outputs, size = [batch_size])]
        random_code = 2 * np.random.rand(batch_size, self.code_size) - 1 # shoud be like after tanh
        randoms = np.concatenate((random_targets, random_code), axis = 1)
        return self.preprocess_code(randoms)

    def preprocess_code(self, code):
        labels, data = code[:, :self.num_outputs], code[:, self.num_outputs:]
        data = np.reshape(data, [-1, self.num_outputs, self.caps_len])
        labels = np.expand_dims(labels, -1)
        return labels * data # masking out all except capsule for label

#
# With Capsules
#
class SemiSupLossConfig():
    def __init__(self, label_smoothing = 0.9):
        self.label_smoothing = label_smoothing
        self.margin_m_plus = label_smoothing
        self.margin_m_minus = 0.0
        self.margin_lambda = 0.9

        self.reconst_coef_g = 0.8
        self.reconst_coef_d = 0.3
        self.gen_classify_coef_g = 0.0
        self.gen_classify_coef_d = 0.0

        self.prediction_coef = 1.0

        self.use_feature_matching = True

class SemiSupCapsConfig():
    def __init__(self, num_outputs = 10, caps1_len = 8, caps2_len = 16, capsg_len = 8, capsg_size = [4, 4, 32], loss_config = SemiSupLossConfig()):
        # discriminator
        self.conv_d = [
                ConvData(32, 5, 2, activation = leaky_relu),
                ConvData(64, 3, 2, activation = leaky_relu),
                ConvData(128, 3, 2, activation = None)
            ]

        self.caps1_len = caps1_len

        self.caps2_len = caps2_len
        self.num_outputs = num_outputs
        self.code_size = caps2_len * num_outputs
        self.leaky_alpha = 0.01

        self.minibatch_num = 32
        self.minibatch_len = 16

        # generator
        self.capsg_len = capsg_len
        self.capsg_size = capsg_size
        self.deconv_g = [
                ConvData(64, 3, 2, activation = leaky_relu),
                ConvData(32, 3, 2, activation = leaky_relu),
                ConvData(3, 5, 2, activation = None)
            ]

        self.use_minibatch = True
        self.with_reconstruction = False
        self.loss_config = loss_config

        self.ae_with_predictions = True

class SemiCapsNet(BasicNet):
    def __init__(self, config = SemiSupCapsConfig()):
        self.ref_batch = None
        self.code_gen = SemiCapsCodeGenerator(config.num_outputs, config.caps2_len)

        modes_dict = {
            'semi-supervised': GanRunner(),
            'gan' : GanRunner(semi_supervised = False),
            'ae' : AERunner(with_predictions = config.ae_with_predictions)
            }
        generator = CapsGenerator(batch_norm = BatchRenormElement())
        generator.set_deconv_block_class(UpsamplingBlockElement)
        element_dict = {
            'generator': generator,
            'encoder': CapsEncoder(),
            'predictor': CapsPredict(),
            'minibatcher': Minibatcher(),
            'labeler': CapsCodePrepare(),
            'extractor': CapsFeatureExtractor(),
            'fake_detector': DenseFakeDetector(),
            }
        super(SemiCapsNet, self).__init__(modes_dict, 'semi-supervised', element_dict, config = config)

    def set_reference_batch(self, ref_batch):
        self.run_elements['generator'].set_ref_batch(ref_batch)

    def get_code_generator(self):
        return self.code_gen


