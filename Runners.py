'''
@author: Dmitry
'''

from Elements import *
from CapsTools import *

class BasicRunner():
    def __init__(self):
        pass

    def run(self, config, elements_dict, *args, **kwargs):
        pass

    # Common losses

    def mean_squared_loss(self, targets, outputs):
        axis = list(range(1, len(list(targets.shape))))
        result = tf.reduce_sum(tf.square(targets - outputs), axis = axis)
        return result

    def get_reconstruction_loss(self, images, reconstructed):
        return self.mean_squared_loss(images, reconstructed)

    def get_loss_with_mask(self, loss, labels_mask = None):
        num = tf.shape(loss)[0]
        total = num
        if labels_mask is not None:
            loss = labels_mask * loss
            num = tf.reduce_sum(labels_mask)
        return tf.cast(total, tf.float32) * tf.reduce_sum(loss) / (num + 1e-6)

    def get_softmax_loss(self, targets, predictions, labels_mask = None):
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = predictions), axis = -1)
        return self.get_loss_with_mask(loss, labels_mask)

    def get_margin_loss(self, loss_config, targets, outputs, targets_mask):
        m_plus = loss_config.margin_m_plus
        m_minus = loss_config.margin_m_minus

        margin_loss = targets * tf.square(m_plus - outputs)
        margin_loss += loss_config.margin_lambda * (1.0 - targets) * tf.square(outputs - m_minus)
        loss = tf.reduce_sum(margin_loss, axis = 1)

        return self.get_loss_with_mask(loss, targets_mask)

    # End common losses

    def loss_function(self, config, *args, **kwargs):
        pass

    def num_classified(self, config, *args, **kwargs):
        pass

    def get_minimizers(self, optimizer, loss):
        print('Default minimizers are used!', tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            return [optimizer.minimize(l) for l in loss]

# Class to use as a runner for GAN and SemiSupervised modes
class GanRunner(BasicRunner):

    def __init__(self, semi_supervised = True):
        self.semi_supervised = semi_supervised

    # TODO: use constants instead of names or do something with dictionary
    def run(self, config, elements_dict, inputs, code, training):
        # remember intended labels for code
        code, self.code_labels = elements_dict['labeler'].run(config, code)

        # generate images from code
        self.generated = elements_dict['generator'].run(config, code, False, training)

        # combine generated images with real ones and run classificator
        print('Generated shape:', self.generated.shape)
        sizes = [tf.shape(inputs)[0], tf.shape(self.generated)[0]]
        self.inputs_code = elements_dict['encoder'].run(config, inputs, False, training)
        self.gen_code = elements_dict['encoder'].run(config, self.generated, True, False)#tanh
        self.codes = tf.concat((self.inputs_code, self.gen_code), axis = 0)
        if self.semi_supervised:
            self.pred_real, self.pred_fake = tf.split(elements_dict['predictor'].run(config, self.codes), sizes)

        # extract features and add minibatch features (if needed)
        features = elements_dict['extractor'].run(config, self.codes)
        features = tf.contrib.layers.flatten(features)
        self.real_features, self.fake_features = tf.split(features, sizes)
        if config.use_minibatch:
            self.real_features = elements_dict['minibatcher'].run(config, self.real_features)
            self.fake_features = elements_dict['minibatcher'].run(config, self.fake_features, reuse = True)
        output_features = tf.concat((self.real_features, self.fake_features), 0)

        output_features = tf.verify_tensor_all_finite(output_features, 'Not all values are finite within minibatch')

        if config.with_reconstruction:
            self.reconstructed = elements_dict['generator'].run(config, self.inputs_code, True, training)
        # discriminate fake and original images
        self.orig, self.fake = tf.split(elements_dict['fake_detector'].run(config, output_features), sizes)

    # GAN losses
    def get_feature_matching_loss(self, real, fake):
        num_features = tf.cast(tf.shape(real)[1], tf.float32)
        return l1norm(tf.reduce_mean(real, 0) - tf.reduce_mean(fake, 0))# / num_features

    def get_loss_on_targets(self, loss_config, targets, outputs, targets_mask):
        return self.get_margin_loss(loss_config, targets, outputs, targets_mask)

    def get_mean_squared_loss_on_targets(self, loss_config, targets, outputs, targets_mask):
        loss = self.mean_squared_loss(targets, outputs)
        print('Loss shape', loss.shape, targets_mask.shape)
        return self.get_loss_with_mask(loss, targets_mask)

    def get_fake_detection_loss(self, loss_config, labels, logits):
        smooth = loss_config.label_smoothing
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = smooth * labels, logits = logits)
        return tf.reduce_sum(loss)

    def loss_function(self, config, targets, labels_info = None, images = None):
        lconfig = config.loss_config

        orig_detection_loss = self.get_fake_detection_loss(lconfig, labels = tf.ones_like(self.orig), logits = self.orig)
        fake_detection_loss = self.get_fake_detection_loss(lconfig, labels = tf.zeros_like(self.fake), logits = self.fake)
        camouflage_loss = self.get_fake_detection_loss(lconfig, labels = tf.ones_like(self.fake), logits = self.fake)
        feature_matching_loss = self.get_feature_matching_loss(self.real_features, self.fake_features)

        # basic GAN discriminator loss
        d_loss = orig_detection_loss + fake_detection_loss
        # basic GAN generator loss
        g_loss = camouflage_loss if not lconfig.use_feature_matching else feature_matching_loss

        additional_losses = (orig_detection_loss, fake_detection_loss)

        if (self.semi_supervised):
            loss_func = self.get_loss_on_targets if lconfig.classification else self.get_mean_squared_loss_on_targets
            loss_on_targets = loss_func(lconfig, targets, self.pred_real, labels_info)
            d_loss += 30.0 * loss_on_targets
            additional_losses += (loss_on_targets,)

            if lconfig.loss_on_gen:
                loss_on_gen = loss_func(lconfig, self.code_labels, self.pred_fake, tf.ones([tf.shape(self.code_labels)[0]], dtype = tf.float32))

                d_loss += lconfig.gen_classify_coef_d * loss_on_gen
                g_loss += lconfig.gen_classify_coef_g * loss_on_gen
                additional_losses += (loss_on_gen,)

        if (config.with_reconstruction):
            reconstruction_loss = self.get_reconstruction_loss(images, self.reconstructed)

            d_loss += lconfig.reconst_coef_d * reconstruction_loss
            g_loss += lconfig.reconst_coef_g * reconstruction_loss
            additional_losses += (reconstruction_loss,)
 
        return (d_loss, g_loss) + additional_losses

    def num_classified(self, config, targets, images):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classified = tf.equal(tf.argmax(self.pred_real, axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)

    def get_minimizers(self, optimizer, loss):
        all_vars = tf.trainable_variables()
        d_vars = [v for v in all_vars if v.name.startswith('discriminator')]
        g_vars = [v for v in all_vars if v.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizer_d = optimizer.minimize(loss[0], var_list = d_vars)
            minimizer_g = optimizer.minimize(loss[1], var_list = g_vars)
        return [minimizer_d, minimizer_g]


# Runner for AutoEncoder network

class AERunner(BasicRunner):

    def __init__(self, with_predictions = False, activation_to_use = (tf.tanh, tanh_cross_entropy)):
        self.with_predictions = with_predictions
        self.final_activation, self.cross_entropy = activation_to_use

    def run(self, config, elements_dict, inputs, training):
        code = elements_dict['encoder'].run(config, inputs, training = training)
        self.code = elements_dict['extractor'].run(config, code)
        if self.with_predictions:
            self.predictions = elements_dict['predictor'].run(config, self.code)
 
        self.reconstructed = elements_dict['generator'].run(config, self.code, training = training)
        self.img = self.final_activation(self.reconstructed)

    def loss_function(self, config, targets, images = None):
        if images == None and not self.with_predictions:
            images = targets
        ae_loss = self.get_reconstruction_loss(images, self.img)
        if (self.with_predictions):
            labels_mask = tf.reduce_sum(targets, axis = -1)
            classification_loss = self.get_softmax_loss(targets, self.predictions, labels_mask)
            ae_loss += config.loss_config.prediction_coef * classification_loss
            ae_loss = (ae_loss, classification_loss)
        return ae_loss

    def num_classified(self, config, targets, images = None):
        if (self.with_predictions):
            labels_mask = tf.reduce_sum(targets, axis = -1)
            classified = tf.equal(tf.argmax(self.predictions, axis = -1), tf.argmax(targets, axis = -1))
            return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)
        else:
            return tf.constant(0)

    def get_minimizers(self, optimizer, loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizer = optimizer.minimize(loss[0])
        return [minimizer]


# Runner for classification tasks

class ClassificationRunner(BasicRunner):
    def __init__(self):
        pass

    def run(self, config, elements_dict, inputs, training):
        self.codes = elements_dict['encoder'].run(config, inputs, False, training)
        #caps1 = squash(reshapeToCapsules(tf.transpose(self.codes[0], [0, 3, 1, 2]), config.caps1_len, axis = 1))
        #caps_layer = CapsLayer(caps1, config.num_outputs, config.caps2_len)
        #caps2 = caps_layer.do_dynamic_routing()
        flattened = tf.reduce_mean(self.codes[0], axis = [1, 2])
        self.predictions = [elements_dict['predictor'].run(config, code) for code in [flattened]]
        #self.doubt = [tf.layers.dense(code, 1, activation = tf.sigmoid) for code in self.codes]
        #self.doubt = tf.squeeze(self.doubt[-1], -1)
        #encoded = tf.contrib.layers.flatten(tf.multiply(caps2, maskForMaxCapsule(self.predictions[0])))
        encoded = flattened

        if config.with_reconstruction:
            self.reconstructed = elements_dict['generator'].run(config, encoded, True, training)
            self.reconstructed = tf.reshape(self.reconstructed, [-1, 32, 32, 3])

    def loss_function(self, config, targets, images = None):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        #print(self.predictions.shape)
        #classification_loss = self.get_margin_loss(config.loss_config, targets, self.predictions[0], labels_mask)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = targets, logits = self.predictions[0])#self.get_margin_loss(config.loss_config, targets, self.codes[0], tf.reduce_sum(targets, axis = -1))#tf.nn.softmax_cross_entropy_with_logits_v2(labels = targets, logits = self.codes[0])
        #coef = tf.stop_gradient(tf.reduce_mean(entropy, axis = -1))
        #entropy = (1.0 - self.doubt) * entropy + coef * self.doubt
        classification_loss = tf.reduce_sum(entropy, axis = -1)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #for rl in reg_losses:
        #    print('Reg loss:', rl)
        retVal = classification_loss

        if reg_losses:
            reg_loss = tf.add_n(reg_losses)#tf.Print(, [reg_losses], message = 'losses', summarize = 100)
            retVal = [retVal + reg_loss, retVal, reg_loss]
        else:
            print('No regularization found')
        
        reconstruction_loss = 0
        if config.with_reconstruction:
            reconstruction_loss = tf.reduce_sum(self.get_reconstruction_loss(images, self.reconstructed))
            reconstruction_loss = config.loss_config.reconstruction_coef * reconstruction_loss
            retVal = [*retVal, reconstruction_loss]
            retVal[0] += reconstruction_loss

        return retVal# + config.loss_config.reconstruction_coef * reconstruction_loss + reg_constant * sum(reg_losses)

    def num_classified(self, config, targets, images = None):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classified = tf.equal(tf.argmax(self.predictions[-1], axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)

    def get_minimizers(self, optimizer, loss):
        updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print(updates)
        #all_vars = tf.trainable_variables()
        #for v in all_vars:
        #    print(v)
        
        with tf.control_dependencies(updates):
            return [optimizer.minimize(loss[0])]
    '''
    def get_minimizers(self, optimizer, loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizers = []
            all_vars = tf.trainable_variables()
            print(all_vars)
            for i, l in enumerate(loss):
                vars = [v for v in all_vars if v.name.startswith('encoder/convLayout%d' % i)]
                print('Vars:', vars)
                minimizers.append(optimizer.minimize(l, var_list = vars))
        return minimizers
    '''
   
   



class RegressionRunner(BasicRunner):
    def __init__(self):
        pass

    def run(self, config, elements_dict, inputs, training):
        self.codes = elements_dict['encoder'].run(config, inputs, False, training)
        self.predictions = elements_dict['predictor'].run(config, self.codes)
        self.pred_real = self.predictions
        #self.doubt = [tf.layers.dense(code, 1, activation = tf.sigmoid) for code in self.codes]
        #self.doubt = tf.squeeze(self.doubt[-1], -1)

    def loss_function(self, config, targets, images = None):
        print(targets.shape, self.predictions.shape)
        regression_loss = tf.reduce_sum(self.mean_squared_loss(targets, self.predictions))
        #regression_loss = tf.Print(regression_loss, [self.predictions[0], targets])
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.00007  # Choose an appropriate one.
        return regression_loss + reg_constant * tf.reduce_sum(reg_losses), regression_loss, reg_constant * tf.reduce_sum(reg_losses)

    def num_classified(self, config, targets, images = None):
        #labels_mask = tf.reduce_sum(targets, axis = -1)
        #classified = tf.equal(tf.argmax(self.predictions[-1], axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(0.0)

    def get_minimizers(self, optimizer, loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            return [optimizer.minimize(loss[0])]

