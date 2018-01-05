'''
@author: Dmitry
'''
from Trainer import *

import pickle as pkl
import time
import numpy as np
import os
import urllib.request

from scipy.io import loadmat

# Like Dataset in Trainer but also returns initial images as outputs
# Can produce random noise vectors to use as an input
class DatasetWithReconstruction(Dataset):
    def __init__(self, base, *args, with_reconstruction = True, with_randoms = False, code_size = 80):
        self.with_randoms = with_randoms
        self.with_reconstruction = with_reconstruction
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
        y = dataset.labels[ind[start:end]]

        if (self.with_reconstruction):
            y = (y, X)

        if (self.with_randoms):
            random_targets = np.eye(self.num_labels)[np.random.randint(0, self.num_labels, size = [batch_size])]
            random_code = 2 * np.random.rand(batch_size, self.code_size) - 1 # shoud be like after tanh
            randoms = np.concatenate((random_targets, random_code), axis = 1)
            X = (X, randoms)
        return X, y

def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = (x - x.min()) * ((max - min) / (255 - x.min())) + min # Keep the braces!
    return x

def unscale(x, feature_range = (-1.0, 1.0)):
    min, max = feature_range
    return ((x - min) * 255 / (max - min)).astype(np.uint8)

class SvhnDataset():
    # leave_labels = -1 means use all labels
    def __init__(self, val_split = 0.4, leave_labels = -1, feature_range = (-1, 1), data_dir = 'data/'):
        self.num_labels = 10
        self.val_split = val_split
        self.leave_labels = leave_labels
        self.feature_range = feature_range
        self.data_dir = data_dir

        self.loadDataset()

    def download_if_needed(self, name):
        datafile = self.data_dir + name
        if not os.path.isfile(datafile):
            print('Downloading:', name)
            urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/' + name, datafile)
            print('Download complete!')

    def preprocess(self, images):
        images = np.rollaxis(images, 3)
        return scale(images, self.feature_range)

    def onehot_y(self, data):
        one_hotter = np.eye(self.num_labels)
        return one_hotter[np.reshape(data['y'] - 1, [-1])]

    def loadDataset(self):
        self.download_if_needed('train_32x32.mat')
        self.download_if_needed('test_32x32.mat')

        trainset = loadmat(self.data_dir + 'train_32x32.mat')
        testset = loadmat(self.data_dir + 'test_32x32.mat')

        train_labels = self.onehot_y(trainset)
        zero_labels = np.zeros_like(train_labels)
        if self.leave_labels != -1:
            train_labels = np.concatenate((train_labels[:self.leave_labels], zero_labels[self.leave_labels:]), axis = 0)

        test_img = self.preprocess(testset['X'])
        test_labels = self.onehot_y(testset)
        l = len(testset['y'])
        val_split = int(l * self.val_split)

        self.train = DatasetBase(self.preprocess(trainset['X']), train_labels)
        self.val = DatasetBase(test_img[:val_split], test_labels[:val_split])
        self.test = DatasetBase(test_img[val_split:], test_labels[val_split:])

        print('Train: inputs - ' + str(self.train.images.shape) + '\t outputs - ' + str(self.train.labels.shape))
        print('Val  : inputs - ' + str(self.val.images.shape) + '\t outputs - ' + str(self.val.labels.shape))
        print('Test : inputs - ' + str(self.test.images.shape) + '\t outputs - ' + str(self.test.labels.shape))


    def get_dataset_for_trainer(self, with_reconstruction = True, with_randoms = False):
        return DatasetWithReconstruction((self.train, self.val, self.test), with_reconstruction = with_reconstruction, with_randoms = with_randoms)



