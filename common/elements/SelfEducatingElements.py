'''
@author: Dmitry
'''

import tensorflow as tf
from common.elements.Elements import *


class SelfEducatingDenseElement(RunElement):
    def __init__(self, num_outputs, dropin = 0.05, threshold = 0.8, decay = 0.99, regularizer = None):
        self.regularizer = regularizer
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.dropin = dropin
        self.decay = decay
        self.inputs = None
        self.outputs = None

    def add_dropin(self, outputs):
        if self.dropin == 0.0:
            return outputs

        randoms = tf.random_uniform(tf.shape(outputs))
        dropins_mask = tf.less(randoms, self.dropin)
        dropins_mask = tf.Print(dropins_mask, [tf.reduce_sum(tf.cast(dropins_mask, tf.float32))], message = 'Dropin cells:')
        outputs = outputs + self.threshold * tf.cast(dropins_mask, tf.float32)
        self.outputs_deleted = self.outputs_deleted * tf.cast(tf.logical_not(dropins_mask), tf.float32)
        return outputs

    def training_step(self, results = None):
        inputs = tf.concat([self.inputs, tf.ones((tf.shape(self.inputs)[0], 1), tf.float32)], axis = 1)
        inputs_squared_norm = l2_squared_norm(inputs, keep_dims = True)
        results_f = tf.expand_dims(tf.cast(results, tf.float32), axis = -1)
        results_wrong = 1.0 - results_f # or logical not on results
        outputs_correct = self.outputs * results_f + self.outputs_deleted * results_wrong
        out_for_proj = self.outputs * results_wrong + self.outputs_deleted * results_f
        out_proj_coef = tf.expand_dims(out_for_proj / inputs_squared_norm, axis = 1)
        print('Results shape:', outputs_correct.shape, 'Inputs shape:', inputs.shape)

        # actual training
        out_transposed = tf.transpose(outputs_correct)
        activated = tf.greater(out_transposed, 0.0)
        activation_matrix = tf.cast(activated, tf.float32)
        # combined way
        inputs_mask = tf.greater(inputs, 0.0)
        inputs_mask_neg = tf.logical_not(inputs_mask)
        inputs_activation = tf.cast(inputs_mask, tf.float32) - tf.cast(inputs_mask_neg, tf.float32)
        #tmp
        inputs_activation = tf.expand_dims(inputs, axis = -1)
        proj = inputs_activation * out_proj_coef
        ortho = tf.concat([self.weights, tf.expand_dims(self.biases, axis = 0)], axis = 0) - proj
        vectors = inputs_activation * tf.expand_dims(outputs_correct, axis = 1)
        print('Ortho shape:', ortho.shape)
        correlations = tf.reduce_mean(vectors, axis = 0)# / tf.shape(results_f)[0]
        correlations_wrong = tf.reduce_mean(ortho, axis = 0)# / tf.shape(results_f)[0]
        print('Cor wrong:', correlations_wrong[:-1].shape)

        # update weights
        updates = self.decay * self.weights + 0.5 * (1.0 - self.decay) * correlations[:-1] + 0.5 * (1.0 - self.decay) * correlations_wrong[:-1]
        updates_norms = tf.sqrt(tf.reduce_sum(tf.square(updates), axis = 1, keepdims = True))
        update = tf.assign(self.weights, updates / updates_norms)
        update = tf.Print(update, [tf.transpose(correlations)], message = 'First input correlations:', summarize = 20)
        update = tf.Print(update, [tf.transpose(correlations_wrong)], message = 'First input correlations wrong:', summarize = 20)
        update = tf.Print(update, [tf.reduce_sum(results_f)], 'Correct results:')
        update_biases = self.decay * self.biases + 0.5 * (1.0 - self.decay) * correlations[-1] + 0.5 * (1.0 - self.decay) * correlations_wrong[-1]
        update2 = tf.assign(self.biases, update_biases)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update2)

    def save_train_info(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def build(self, input, dropout = 0.0, training = True, name = 'se_dense', regularizer = None):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            inputs = tf.layers.flatten(input)
            #print('Clever dense inputs shape flattened:', inputs.shape)
            self.weights = tf.get_variable('weights', [inputs.shape[1], self.num_outputs], trainable = False)#, initializer=tf.zeros_initializer)
            self.biases = tf.get_variable('biases', [self.num_outputs], trainable = False, initializer=tf.zeros_initializer)
            # outputs as in normal dense without bias
            outputs = tf.matmul(inputs, self.weights) + self.biases
            self.logits = outputs
            # remove small activations
            masked = tf.greater(outputs, self.threshold)
            #print('Clever dense mask shape:', masked.shape)
            #masked = tf.Print(masked, [tf.shape(outputs)[1], tf.shape(outputs)[0]], message = 'Total cells and number of images:')
            masked = tf.Print(masked, [inputs], 'Input:', summarize = 4)
            masked = tf.Print(masked, [tf.reduce_sum(tf.cast(masked, tf.float32))], message = 'Activated cells:')
            result = outputs * tf.cast(masked, tf.float32)
            self.outputs_deleted = outputs - result
            result = tf.cond(training, lambda: self.add_dropin(result), lambda: result)
            self.save_train_info(inputs, result)
            # training step should be done somewhere!!!
            #print('Clever dense result shape after cond:', result.shape)
            result = tf.maximum(tf.minimum(result, 1.0), 0.0)
            result = tf.Print(result, [tf.transpose(self.weights), self.biases], message = 'First input cell weights:', summarize = 16)
        return result
