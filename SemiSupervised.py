'''
@author: Dmitry
'''
from Trainer import *
from SemiSupervisedNet import *

import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

data_dir = 'data/'

# !!! Requires data to be in data_dir
trainset = loadmat(data_dir + 'train_32x32.mat')
testset = loadmat(data_dir + 'test_32x32.mat')

def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = (x - x.min()) * ((max - min) / (255 - x.min())) + min # Braces are important! Somewhy...
    return x

def preprocess(images):
    images = np.rollaxis(images, 3)
    return scale(images)

print('Train:' + str(trainset['X'].shape) + '\t Test:' + str(testset['X'].shape))
print('Labels Train:' + str(trainset['y'].shape) + '\t Test:' + str(testset['y'].shape))
leave_labels = 1000
num_labels = 10

one_hotter = np.eye(num_labels)
train_labels = one_hotter[np.reshape(trainset['y'] - 1, [-1])]
zero_labels = np.zeros_like(train_labels)
train_labels = np.concatenate((train_labels[:leave_labels], zero_labels[leave_labels:]), axis = 0)

test_img = preprocess(testset['X'])
test_labels = one_hotter[np.reshape(testset['y'] - 1, [-1])]
l = len(testset['y'])
split_coef = 0.4
val_split = int(l * split_coef)

train = DatasetBase(preprocess(trainset['X']), train_labels)
val = DatasetBase(test_img[:val_split], test_labels[:val_split])
test = DatasetBase(test_img[val_split:], test_labels[val_split:])

class AE_Dataset(Dataset):
    def __init__(self, base, *args, with_randoms = False, code_size = 80):
        self.with_randoms = with_randoms
        self.num_labels = len(base[0].labels[0])
        self.code_size = code_size
        Dataset.__init__(self, base, *args)
        print('num_labels:', self.num_labels)

    def get_batch(self, dataset, num, batch_size, shuffle = True):
        start, end = batch_size * num, batch_size * (num + 1)
        ind = np.arange(len(dataset.images))
        if (shuffle):
            np.random.shuffle(ind)

        X = dataset.images[ind[start:end]]
        y1 = dataset.labels[ind[start:end]]
        y2 = X
        if (self.with_randoms):
            random_targets = np.eye(self.num_labels)[np.random.randint(0, self.num_labels, size = [batch_size])]
            random_code = 2 * np.random.rand(batch_size, self.code_size) - 1 # shoud be like after tanh
            randoms = np.concatenate((random_targets, random_code), axis = 1)
            X = (X, randoms)
        return X, (y1, y2)

mode = 'semi'
base_folder = 'semi-supervised'
batch_size = 1000

need_resave = False
dataset = AE_Dataset((train, val, test))
network_base = SemiSupervisedNetwork()
params = TrainerParams()
params.batch_size = batch_size

if (mode == 'ae' or mode == 'both'):
    tf.reset_default_graph()
    network = Network(network_base, 'ae_loss', 'run_autoencoder', 'num_classified_ae', minimizer = 'get_minimizers_ae')
    trainer = Trainer(network, dataset, params)
    saver = CustomSaver(folders=[base_folder + '/ae', base_folder + '/ae/epoch'])
    trainer.train(saver)
    need_resave = True

if (mode == 'semi' or mode == 'both'):
    tf.reset_default_graph()
    dataset.with_randoms = True
    network2 = Network(network_base, 'semi_loss', 'run_semi_supervised', 'num_classified_semi', 'get_minimizers_semi')
    trainer2 = Trainer(network2, dataset, params)
    saver2 = CustomSaver(folders=[base_folder + '/semi', base_folder + '/semi/epoch'])
    if (need_resave):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver2.restore_session(sess, filename = saver2.get_last_saved(base_folder + '/ae'))
            saver2.save_session(sess, False, (0, 0), (0, [float('inf'), float('inf')], 0))
            print('Resaved!')
    trainer2.train(saver2)

def unscale(x, scaling = (-1.0, 1.0)):
    min, max = scaling
    return ((x - min)*255 / (max - min)).astype(np.uint8)

'''
tf.reset_default_graph()
trainer = Trainer(network2, dataset2, params)
last_saved = tf.train.latest_checkpoint(base_folder + '/semi')
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
z = tf.placeholder(tf.float32, [None, 90])

with tf.Session() as sess:
    saver.restore_session(sess)
    (images, randoms), _ = dataset2.get_batch(dataset.val, 0, 6, False)
    
    #images2 = np.rollaxis(testset['X'], 3)[:6]
    #images2 = sess.run(network_base.img, feed_dict={trainer.input_data: images, trainer.training: False})
    images2 = sess.run(network_base.generated, feed_dict={trainer.input_data: (images, randoms), trainer.training: False})
    images = np.concatenate((images, images2), axis = 0)
    images = unscale(images)
    fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12,3),)
    for ii, ax in zip(images, axes.flatten()):
        ax.axis('off')
        
        ax.set_adjustable('box-forced')
        ax.imshow(ii)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
'''