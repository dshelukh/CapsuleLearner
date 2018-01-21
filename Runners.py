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

    def get_reconstruction_loss(self, images, reconstructed):
        return tf.reduce_sum(tf.reduce_mean(tanh_cross_entropy(logits = reconstructed, labels = images), axis = [-1, -2, -3]))

    def get_softmax_loss(self, targets, predictions, labels_mask = None):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = predictions), axis = -1)
        if labels_mask is not None:
            loss = labels_mask * loss
        return tf.reduce_sum(loss)

    # End common losses

    def loss_function(self, config, *args, **kwargs):
        pass

    def num_classified(self, config, *args, **kwargs):
        pass

    def get_minimizers(self, optimizer, loss):
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
        images = tf.concat((inputs, tf.tanh(self.generated)), axis = 0)
        sizes = [tf.shape(inputs)[0], tf.shape(self.generated)[0]]
        self.codes = elements_dict['encoder'].run(config, images, False, training)
        if self.semi_supervised:
            self.pred_real, self.pred_fake = tf.split(elements_dict['predictor'].run(config, self.codes), sizes)
        self.inputs_code, self.gen_code = tf.split(self.codes, sizes)

        # extract features and add minibatch features (if needed)
        features = elements_dict['extractor'].run(config, self.codes)
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
        m_plus = loss_config.margin_m_plus
        m_minus = loss_config.margin_m_minus

        margin_loss = targets * tf.square(m_plus - outputs)
        margin_loss += loss_config.margin_lambda * (1.0 - targets) * tf.square(outputs - m_minus)
        loss = targets_mask * tf.reduce_sum(margin_loss, axis = 1)

        return (tf.to_float(tf.shape(targets)[0]) * tf.reduce_sum(loss)) / (tf.reduce_sum(targets_mask) + 1e-6)

    def get_fake_detection_loss(self, loss_config, labels, logits):
        smooth = loss_config.label_smoothing
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = smooth * labels, logits = logits)
        return tf.reduce_sum(loss)

    def loss_function(self, config, targets, images):
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
            loss_on_targets = self.get_loss_on_targets(lconfig, targets, self.pred_real, tf.reduce_sum(targets, axis = -1))
            loss_on_gen = self.get_loss_on_targets(lconfig, self.code_labels, self.pred_fake, tf.ones([tf.shape(self.code_labels)[0]], dtype = tf.float32))

            d_loss += lconfig.gen_classify_coef_d * loss_on_gen + loss_on_targets
            g_loss += lconfig.gen_classify_coef_g * loss_on_gen
            additional_losses += (loss_on_targets, loss_on_gen)

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

    def __init__(self, with_predictions = False):
        self.with_predictions = with_predictions

    def run(self, config, elements_dict, inputs, training):
        self.code = elements_dict['encoder'].run(config, inputs, training = training)
        if self.with_predictions:
            self.predictions = elements_dict['predictor'].run(config, self.codes)
 
        self.reconstructed = elements_dict['generator'].run(config, self.code, training = training)
        self.img = tf.tanh(self.reconstructed)

    def loss_function(self, config, targets, images):
        ae_loss = self.get_reconstruction_loss(images, self.reconstructed)
        if (self.with_predictions):
            labels_mask = tf.reduce_sum(targets, axis = -1)
            classification_loss = self.get_softmax_loss(targets, self.predictions, labels_mask)
            ae_loss += config.loss_config.prediction_coef * classification_loss
        return ae_loss

    def num_classified(self, config, targets, images):
        if (self.with_predictions):
            labels_mask = tf.reduce_sum(targets, axis = -1)
            classified = tf.equal(tf.argmax(self.predictions, axis = -1), tf.argmax(targets, axis = -1))
            return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)
        else:
            return tf.constant(0)


