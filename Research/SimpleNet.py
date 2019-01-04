'''
@author: Dmitry
'''

import sys
sys.path.insert(0, '../')
from SvhnDataset import SvhnDataset
from CifarDataset import Cifar10Dataset


from os import listdir
from os.path import isfile, join

from Trainer import *
import numpy as np
from SemiSupervisedNet import *
import gc

class SimpleDataset(Dataset):
    def __init__(self, *args, num = 2, num_labels = 16, batch_num = 20):
        Dataset.__init__(self, ({}, {}, {}), *args)
        self.num = num
        self.num_labels = num_labels
        self.batch_num = batch_num
        self.corner_mean = 0.9
        self.corner_std = 0.05
        self.other_mean = 0.3
        self.other_std = 0.15

    def onehot_y(self, data):
        one_hotter = np.eye(self.num_labels)
        return one_hotter[np.reshape(data - 1, [-1])]

    def get_batch(self, dataset, num, batch_size, shuffle = True):
        X = []
        y = []
        for i in range(batch_size):
            example = []
            result = 0
            d = 0
            om, cm = self.other_mean, self.corner_mean
            if (np.random.rand() < 0.5):
                om, cm = cm, om
            for j in range(self.num):
                row = []
                for k in range(self.num):
                    val = np.random.normal(om, self.other_std)
                    if (j == 0 or j == self.num - 1) and (k == 0 or k == self.num - 1):
                        if (np.random.random() > 0.5):
                            val = np.random.normal(cm, self.corner_std)
                            result += 2 ** d
                        d += 1
                    row.append([val])
                example.append(row)
            X.append(example)
            y.append(result)
        return np.array(X), self.onehot_y(np.array(y))

    def get_num_batches(self, _, batch_size):
        return self.batch_num

    def get_dataset(self, name):
        return []

    def get_size(self, dataset):
        return self.batch_num * 1000 # batch_size

class SimpleLossConfig():
    def __init__(self):
        self.margin_m_plus = 0.9
        self.margin_m_minus = 0.1
        self.margin_lambda = 0.5
        
        self.reconstruction_coef = 0.0005

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
        self.batch_norm = BatchNormElement
        self.dropout = 0.3
        self.weight_decay = 0.0005
        '''
        self.save_name = 'save_weirdnet0.1'
        self.convs = ConvLayout([
                #ConvData(4, (2, 2), (1, 1), activation = tf.nn.relu)
                #ConvData(32, (5, 5), 2, activation = tf.nn.relu, element = WeirdConvBlockElement),
                #ConvData(64, (5, 5), 2, activation = tf.nn.relu, element = WeirdConvBlockElement),
                
                #ConvLayout([
                #    ConvData(32, (9, 9), 4, activation = tf.nn.relu, element = self.conv_element),
                #    ConvLayout([
                #        ConvData(16, (5, 5), 2, activation = tf.nn.relu, element = self.conv_element),
                #        ConvData(32, (5, 5), 2, activation = tf.nn.relu, element = self.conv_element),
                #        ]),
                #    ConvLayout([
                #        ConvData(16, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                #        ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                #        ConvData(32, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                #        ]),
                #    ConvLayout([
                #        ConvData(16, (7, 1), (2, 1), activation = tf.nn.relu, element = self.conv_element),
                #        ConvData(16, (1, 7), (1, 2), activation = tf.nn.relu, element = self.conv_element),
                #        ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                #        ]),
                #    ], isParallel = True),
                #ConvData(128, (5, 5), 2, activation = tf.nn.relu, element = self.conv_element),
                #ConvData(256, (5, 5), 2, activation = tf.nn.relu, element = self.conv_element)
                
                ConvLayout([
                    ConvData(16, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                    ConvData(8, (3, 3), 1, activation = tf.sigmoid, element = WeirdIdealConvBlockElement),
                    ConvData(8, (3, 3), 1, activation = lambda x:1.0 - tf.sigmoid(x), element = WeirdIdealConvBlockElement),
                    ], isParallel = True),
                BatchNormBlock(element = self.batch_norm),
                ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                DropoutBlock(self.dropout),
                ConvLayout([
                    ConvData(48, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                    ConvData(8, (3, 3), 1, activation = tf.sigmoid, element = WeirdIdealConvBlockElement),
                    ConvData(8, (3, 3), 1, activation = lambda x:1.0 - tf.sigmoid(x), element = WeirdIdealConvBlockElement),
                    ], isParallel = True),
                BatchNormBlock(element = self.batch_norm),
                ConvData(64, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                DropoutBlock(self.dropout),
                ConvLayout([
                    ConvData(32, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
                    ConvData(16, (3, 3), 1, activation = tf.sigmoid, element = WeirdIdealConvBlockElement),
                    ConvData(16, (3, 3), 1, activation = lambda x:1.0 - tf.sigmoid(x), element = WeirdIdealConvBlockElement),
                    ], isParallel = True),
                BatchNormBlock(element = self.batch_norm),
                ConvData(128, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                DropoutBlock(self.dropout),
                ConvData(128, (1, 1), 1, activation = lambda x:1.0 - tf.sigmoid(x), element = WeirdIdealConvBlockElement),
                BatchNormBlock(element = self.batch_norm),
                ConvData(128, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
                DropoutBlock(self.dropout),
            ])
        '''
        '''
        self.save_name = 'save_simplenet0.1'
        self.convs = ConvLayout([
            ConvData(32, (3, 3), 1, activation = tf.sigmoid, element = self.conv_element),
            DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            ConvData(32, (3, 3), 2, activation = tf.sigmoid, element = self.conv_element),
            DropoutBlock(self.dropout),
            ConvData(32, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
            DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 1, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            #ConvData(128, (1, 1), 1, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            #ConvData(128, (3, 3), 2, activation = tf.nn.relu, element = self.conv_element),
            #DropoutBlock(self.dropout),
            ], batch_norm = self.batch_norm())
        '''
        self.scale = 4
        
        
        self.save_name = 'save_wrn_0.13'
        self.convs = ConvLayout([
            ConvData(16, (3, 3), 1, element = self.conv_element),#WeirdConvBlockElement_),
            BatchNormBlock(element = self.batch_norm),
            ActivationBlock(tf.nn.relu),
            #PreliminaryResultElement(self.num_outputs, 0.95, False, self.dense_element),

#            BasicResBlock(16 * self.scale, 1, dropout = self.dropout, element = self.conv_element, adjust_input=True, start_with_activation = False),
#            BasicResBlock(16 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
            #BasicResBlock(16 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BatchNormBlock(element = self.batch_norm),
#            ActivationBlock(tf.nn.relu),
            #PreliminaryResultElement(self.num_outputs, 0.9, True, self.dense_element),

#            BasicResBlock(32 * self.scale, 2, dropout = self.dropout, element = self.conv_element, adjust_input=True, start_with_activation = False),
#            BasicResBlock(32 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
            #BasicResBlock(32 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BatchNormBlock(element = self.batch_norm),
#            ActivationBlock(tf.nn.relu),
            #PreliminaryResultElement(self.num_outputs, 0.90, True, self.dense_element),

#            BasicResBlock(64 * self.scale, 2, dropout = self.dropout, element = self.conv_element, adjust_input=True, start_with_activation = False),
#            BasicResBlock(64 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
            #BasicResBlock(64 * self.scale, 1, dropout = self.dropout, element = self.conv_element),
#            BatchNormBlock(element = self.batch_norm),
#            ActivationBlock(tf.nn.relu),

            FinalizingElement(self.num_outputs, False, self.dense_element)
            ], batch_norm = EmptyElement())
        
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
            'predictor': DensePredict()#CapsPredict()#EmptyElementConfig()
            }
        super(SimpleNetwork, self).__init__(modes_dict, 'classification', element_dict, config = config)



def augment_data(data, max_translate = (2, 2)):
    mtx, mty = max_translate
    N = tf.shape(data)[0]
    transform_mat = tf.concat([tf.ones([N, 1]), tf.zeros([N, 1]), tf.random_uniform([N, 1], minval = -mtx, maxval = mtx),
                               tf.zeros([N, 1]), tf.ones([N, 1]), tf.random_uniform([N, 1], minval = -mty, maxval = mty),
                               tf.zeros([N, 1]), tf.zeros([N, 1])], 1)
    return tf.contrib.image.transform(data, transform_mat)

class CappedOptimizer():
    def __init__(self, x):
        self.optimizer = tf.train.MomentumOptimizer(x, 0.9, use_nesterov=True)

    def minimize(self, loss):
        gvs = self.optimizer.compute_gradients(loss)
        #g, v = gvs[0]
        #print(g.shape, tf.is_nan(g).shape, tf.clip_by_value(tf.boolean_mask(g, tf.is_nan(g)), -1., 1.).shape)
        capped_gvs = [(tf.clip_by_value(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), -10., 10.), var) for grad, var in gvs]
        return self.optimizer.apply_gradients(capped_gvs)

config = SimpleNetworkConfig()
#dataset = SvhnDataset(val_split=0.0, feature_range = (0, 1), with_extra = True).get_dataset_for_trainer(with_reconstruction = False)
#dataset = SimpleDataset(num = 3, num_labels = config.num_outputs)
dataset = Cifar10Dataset(feature_range = (0, 1)).get_dataset_for_trainer(with_reconstruction = False)
gc.collect()



save_folder = config.save_name

params = TrainerParams()
params.batch_size = 128
params.val_check_period = 750
params.max_epochs = 160
params.early_stopping = False

params.optimizer = lambda x: tf.train.MomentumOptimizer(x, 0.9, use_nesterov=True)#lambda x: CappedOptimizer(x)#lambda x: tf.train.MomentumOptimizer(x, 0.9, use_nesterov=True)#lambda x: tf.train.AdamOptimizer(x, 0.92)
params.learning_rate = EpochListScheduler(0.001, [(40, 0.01), (60, 0.001), (90, 0.0001)])#EpochScheduler(0.01, 0.1, 80)
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




