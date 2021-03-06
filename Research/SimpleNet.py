'''
@author: Dmitry
'''

import sys
sys.path.insert(0, '../')
from common.dataset.SvhnDataset import SvhnDataset
from common.dataset.CifarDataset import Cifar10Dataset


from os import listdir
from os.path import isfile, join

from common.Trainer import *
from common.elements.Elements import *
from common.network.NetworkBase import *
from Runners import *

import numpy as np
import gc

class SimpleLossConfig():
    def __init__(self):
        self.margin_m_plus = 0.9
        self.margin_m_minus = 0.1
        self.margin_lambda = 0.5
        
        self.reconstruction_coef = 0.002

def BasicResBlock(num_features, first_stride = 1, batch_norm = BatchNormElement, element = WeirdConvBlockElement, dropout = 0.4, adjust_input = False, start_with_activation = True):
    input_conv = None if (not adjust_input) and (first_stride == 1) else ConvData(int(num_features), (1, 1), first_stride, element = element)#WeirdConvBlockElement)

    start = [] if not start_with_activation else [BatchNormBlock(element = batch_norm), ActivationBlock(tf.nn.relu)]
    return ResidualBlock(ConvLayout([
                *start,
                ConvData(int(num_features), (3, 3), first_stride, element = element),

                BatchNormBlock(element = batch_norm),
                ActivationBlock(tf.nn.relu),
                DropoutBlock(dropout),
                ConvData(int(num_features), (3, 3), 1, element = element)]),
                          input_conv)
        
class SimpleNetworkConfig():
    def __init__(self, num_outputs = 10):
        self.num_outputs = num_outputs
        self.conv_element = ConvBlockElement
        self.dense_element = DenseElement
        self.batch_norm = BatchRenormElement
        self.dropout = 0.3
        self.weight_decay = 0.0005

        self.scale = 4
        
        
        self.save_name = 'save_convs'
        self.convs = ConvLayout([
            #ReshapeBlock([3072]),
            #DenseBlockElement(1024),
            #CleverDenseElement(1024, dropin = 0.01, decay = 0.999),
            #ReshapeBlock([32, 32, 1])
            ConvData(16, (3, 3), 1, element = self.conv_element),#WeirdConvBlockElement_),
            BatchNormBlock(element = self.batch_norm),
            ActivationBlock(tf.nn.relu),
            #PreliminaryResultElement(self.num_outputs, 0.35, False, self.dense_element),

            BasicResBlock(16 * self.scale, 1, dropout = self.dropout, element = self.conv_element, adjust_input=True, start_with_activation = False),
#            BasicResBlock(16 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BasicResBlock(16 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BatchNormBlock(element = self.batch_norm),
#            ActivationBlock(tf.nn.relu),
            #PreliminaryResultElement(self.num_outputs, 0.4, True, self.dense_element),

#            BasicResBlock(32 * self.scale, 2, dropout = self.dropout, element = self.conv_element, adjust_input=True, start_with_activation = False),
#            BasicResBlock(32 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BasicResBlock(32 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BatchNormBlock(element = self.batch_norm),
#            ActivationBlock(tf.nn.relu),
            #PreliminaryResultElement(self.num_outputs, 0.5, True, self.dense_element),

#            BasicResBlock(64 * self.scale, 2, dropout = self.dropout, element = ConvBlockElement, adjust_input=True, start_with_activation = False),
#            BasicResBlock(64 * self.scale, 1, dropout = self.dropout, element = ConvBlockElement),
#            BasicResBlock(64 * self.scale, 1, dropout = self.dropout, element = ConvBlockElement),
#            BatchNormBlock(element = self.batch_norm),
#            ActivationBlock(tf.nn.relu),

            FinalizingElement(self.num_outputs, True, self.dense_element)
            ], batch_norm = EmptyElement())
        
        self.dense_g = [
            (8 * 8 * 3, tf.sigmoid),
            #(8 * 8 * 3 * 2, tf.sigmoid),
            (8 * 8 * 3 * 4, tf.sigmoid),
            #(8 * 8 * 3 * 4 * 2, tf.sigmoid),
            (8 * 8 * 3 * 4 * 4, tf.sigmoid),
            ]

        self.weirds = [
            (4, ConvLayout([])),
            (8, ConvLayout([
                ConvData(8, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                ConvData(8, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ])),
            (8, ConvLayout([
                ConvData(8, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element)
                ])),
            (12, ConvLayout([
                ConvData(12, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                ConvData(12, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ])),
            (16, ConvLayout([
                ConvData(16, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                ConvData(16, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ConvData(16, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ])),
            (16, ConvLayout([
                ConvData(16, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                ConvData(16, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ConvData(16, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ])),
            (32, ConvLayout([
                ConvData(32, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ])),
            (32, ConvLayout([
                ConvData(32, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ConvData(64, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                ])),
            ]
        self.conv_d = self.convs
        self.with_reconstruction = False
        self.loss_config = SimpleLossConfig()
        self.caps1_len = 8

        self.caps2_len = 16

class SimpleNetwork(BasicNet):
    def __init__(self, config = SimpleNetworkConfig()):
        modes_dict = {
            'classification' : ClassificationRunner()
            }
        element_dict = {
            'encoder': ConvEncoder(config.convs, DenseElement), #CapsEncoder(),#ConvEncoder(),
            'predictor': EmptyElementConfig(),#DensePredict(),#CapsPredict(),#EmptyElementConfig(),
            'generator': DenseEncoder(config.dense_g, 'generator'), #CapsEncoder(),#ConvEncoder(),
            }
        super(SimpleNetwork, self).__init__(modes_dict, 'classification', element_dict, config = config)


class CappedOptimizer():
    def __init__(self, x):
        self.optimizer = tf.train.MomentumOptimizer(x, 0.9, use_nesterov=True)

    def minimize(self, loss):
        gvs = self.optimizer.compute_gradients(loss)
        #g, v = gvs[0]
        #print(g.shape, tf.is_nan(g).shape, tf.clip_by_value(tf.boolean_mask(g, tf.is_nan(g)), -1., 1.).shape)
        capped_gvs = [(tf.clip_by_value(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), -1., 1.), var) for grad, var in gvs]
        return self.optimizer.apply_gradients(capped_gvs)

config = SimpleNetworkConfig()
#dataset = SvhnDataset(val_split=0.0, feature_range = (0, 1), with_extra = False).get_dataset_for_trainer(with_reconstruction = config.with_reconstruction)
#dataset = SimpleDataset(num = 3, num_labels = config.num_outputs)
dataset = (
    SvhnDataset()
    .get_dataset_for_trainer(with_reconstruction = config.with_reconstruction)
    )


save_folder = config.save_name

params = TrainerParams()
params.batch_size = 128
params.val_check_period = 0
params.max_epochs = 170
params.early_stopping = False

#params.optimizer = lambda x: tf.train.MomentumOptimizer(x, 0.9, use_nesterov=True)
params.optimizer = lambda x: CappedOptimizer(x)
#params.optimizer = lambda x: tf.train.AdamOptimizer(x, 0.92)


params.learning_rate = EpochListScheduler(0.03, [(60, 0.01), (120, 0.003), (160, 0.001)])#EpochScheduler(0.01, 0.1, 80)
#config.weight_decay = config.weight_decay / params.batch_size
network_base = SimpleNetwork(config)
network = Network(network_base, *network_base.get_functions_for_trainer())

tf.reset_default_graph()

trainer = Trainer(network, dataset, params)

saver = CustomSaver(folders=[save_folder + '/ae', save_folder + '/ae/epoch'])
trainer.train(saver, restore_from_epochend = True)#, augmentation = augment_data)

'''
print(dir())
import matplotlib.pyplot as plt
with tf.Session() as sess:
    batch, _ = dataset.get_batch(dataset.get_dataset('train'), 0, 6)
    data = tf.placeholder(tf.float32, [None, 32, 32, 3])
    result = sess.run(augment_data(data), feed_dict={data: batch})
    images = np.concatenate((batch, result), axis = 0)
    images = unscale(images)
    fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12,3),)
    for ii, ax in zip(images, axes.flatten()):
        ax.axis('off')
        ax.imshow(ii)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
'''
print('Finished SimpleNet')




