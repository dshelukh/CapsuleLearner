'''
@author: Dmitry
'''

import tensorflow as tf

from Trainer import *
import code

class ConvData():
    def __init__(self, num_features, kernel, stride, padding = 'same'):
        self.num_features, self.kernel, self.stride = num_features, kernel, stride
        self.padding = padding

    def get_conv_data(self):
        return self.num_features, self.kernel, self.stride, self.padding

def leaky_relu(x, alpha = 0.01):
    return tf.maximum(x, alpha * x)

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

def tanh_cross_entropy(logits, labels):
    abs_logits = tf.abs(logits)
    return abs_logits + logits * labels + tf.log(1 + tf.exp(-2.0 * abs_logits))

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




#
# With Capsules
#
class SemiSupCapsConfig():
    def __init__(self, num_outputs = 10, caps1_len = 8, caps2_len = 16, capsg_len = 8, capsg_size = [8, 8, 32]):
        # discriminator
        self.conv1_info = ConvData(128, 3, 2)
        self.conv2_info = ConvData(128, 3, 2)
        self.caps1_len = caps1_len

        self.caps2_len = caps2_len
        self.num_outputs = num_outputs
        self.code_size = caps2_len * num_outputs
        self.leaky_alpha = 0.01

        # generator
        self.capsg_len = capsg_len
        self.capsg_size = capsg_size
        self.deconv1_info = ConvData(128, 3, 2)
        self.deconv2_info = ConvData(3, 3, 2)

class SemiSupCapsNet():
    def __init__(self, config = SemiSupCapsConfig()):
        self.config = config

    def run_encoder(self, inputs, reuse = False, training = False):
        config = self.config

        with(tf.variable_scope('discriminator', reuse=reuse)):
            conv1_info = config.conv1_info
            conv1 = tf.layers.conv2d(inputs, conv1_info.num_features, conv1_info.kernel, conv1_info.stride, padding = 'same')
            conv1 = tf.layers.batch_normalization(conv1, training = training)
            conv1 = leaky_relu(conv1, config.leaky_alpha)

            conv2_info = config.conv2_info
            conv2 = tf.layers.conv2d(conv1, conv2_info.num_features, conv2_info.kernel, conv2_info.stride, padding = 'same')
            #conv2 = tf.layers.batch_normalization(conv2, training = training) # do we need it?

            caps1 = squash(reshapeToCapsules(tf.transpose(conv2, [0, 3, 1, 2]), config.caps1_len, axis = 1))
            caps_layer = CapsLayer(caps1, config.num_outputs, config.caps2_len)
            caps2 = caps_layer.do_dynamic_routing()
            return caps2

    def get_masked_code(self, code):
        masked = tf.multiply(code, maskForMaxCapsule(norm(code)))
        return masked # tf.contrib.layers.flatten(masked)

    def generate_from_code(self, masked_code, reuse = False, training = True):
        config = self.config

        with(tf.variable_scope('generator', reuse=reuse)):
            caps = squash(masked_code)
            caps_layer = CapsLayer(caps, np.prod(config.capsg_size), config.capsg_len)
            caps2 = caps_layer.do_dynamic_routing()
            reshaped = tf.reshape(caps2, [-1, config.capsg_size[0], config.capsg_size[1], config.capsg_size[2] * config.capsg_len])

            deconv1_info = config.deconv1_info
            gconv1 = tf.layers.conv2d_transpose(reshaped, *(deconv1_info.get_conv_data()))
            gconv1 = tf.layers.batch_normalization(gconv1, training = training)
            gconv1 = leaky_relu(gconv1, config.leaky_alpha)

            deconv2_info = config.deconv2_info
            gconv2 = tf.layers.conv2d_transpose(gconv1, *(deconv2_info.get_conv_data()))
            return gconv2

    def is_fake(self, code, reuse = False):
        with (tf.variable_scope('discriminator_add', reuse = reuse)):
            flattened = tf.contrib.layers.flatten(code)
            fake_detector = tf.layers.dense(flattened, 1)
        return fake_detector

    def run(self, inputs, code, training = True):
        config = self.config

        code = code[:, config.num_outputs:]
        caps_code = tf.reshape(code, [-1, config.num_outputs, config.caps2_len])
        self.code_labels = tf.one_hot(tf.argmax(norm(caps_code), axis = -1), config.num_outputs)
        self.generated = self.generate_from_code(caps_code, False, training)

        images = tf.concat((inputs, tf.tanh(self.generated)), axis = 0)
        sizes = [tf.shape(inputs)[0], tf.shape(self.generated)[0]]
        self.codes = self.run_encoder(images, False, training)

        self.inputs_code, self.gen_code = tf.split(self.codes, sizes)
        self.reconstructed = self.generate_from_code(self.inputs_code, True, training)
        self.orig, self.fake = tf.split(self.is_fake(self.codes), sizes)

    def loss_function(self, targets, images):
        targets_mask = tf.reduce_sum(targets, axis = -1)

        loss_on_targets = targets_mask * tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = norm(self.inputs_code))
        loss_on_gen = tf.nn.softmax_cross_entropy_with_logits(labels = self.code_labels, logits = norm(self.gen_code))
        reconstruction_loss = tanh_cross_entropy(logits = self.reconstructed, labels = images)
        orig_detection_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.orig), logits = self.orig)
        fake_detection_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.fake), logits = self.fake)
        fake_error_detection_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.fake), logits = self.fake)
        
        d_loss = tf.reduce_sum(loss_on_targets) + 0.0001 * tf.reduce_sum(reconstruction_loss) #+ 0.01 * tf.reduce_sum(loss_on_gen)
        d_loss += tf.reduce_sum(orig_detection_loss) + tf.reduce_sum(fake_detection_loss)

        g_loss = 0.0001 * tf.reduce_sum(reconstruction_loss) # + 0.01 * tf.reduce_sum(loss_on_gen)
        g_loss += tf.reduce_sum(fake_error_detection_loss)
        return d_loss, g_loss

    def get_num_classified(self, targets, labels):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classified = tf.equal(tf.argmax(labels, axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)

    def num_classified(self, targets, images):
        return self.get_num_classified(targets, norm(self.inputs_code))

    def get_minimizers(self, optimizer, loss):
        all_vars = tf.trainable_variables()
        d_vars = [v for v in all_vars if v.name.startswith('discriminator')]
        g_vars = [v for v in all_vars if v.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizer_d = optimizer.minimize(loss[0], var_list = d_vars)
            minimizer_g = optimizer.minimize(loss[1], var_list = g_vars)
            #minimizer = optimizer.optimize(loss[0], loss[1], d_vars, g_vars)
        return [minimizer_d, minimizer_g]

    def get_functions_for_trainer(self):
        return 'loss_function', 'run', 'num_classified', 'get_minimizers'

