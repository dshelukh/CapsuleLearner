'''
@author: Dmitry
'''
from DatasetBase import *

import pickle as pkl
import time
import numpy as np
import os
import urllib.request

from scipy.io import loadmat
import sys

#Preprocess from .mat files
class SvhnPreprocessor():
    def __init__(self, num_labels = 10, feature_range = (-1, 1)):
        self.num_labels = num_labels
        self.feature_range = feature_range

    def preprocess_X(self, images):
        #images = np.rollaxis(images, 3)
        return scale(images, self.feature_range).astype(np.float32)

    def onehot_y(self, data):
        one_hotter = np.eye(self.num_labels)
        return one_hotter[np.reshape(data - 1, [-1])]

    def preprocess(self, imgs, labels):
        if imgs.size > 0 and labels.size > 0:
            labels = self.onehot_y(labels)
            imgs = self.preprocess_X(imgs)
        return imgs, labels

class SvhnDataset(DownloadableDataset):
    # leave_labels = -1 means use all labels

    def __init__(self, val_split = 0.3, leave_labels = -1, feature_range = (-1, 1), data_dir = 'data/', with_extra = False):
        self.val_split = val_split
        self.leave_labels = leave_labels
        self.prep = SvhnPreprocessor(feature_range = feature_range)

        self.with_extra = with_extra
        files_to_download = ['train_32x32.mat', 'test_32x32.mat']
        if (self.with_extra):
            files_to_download.append('extra_32x32.mat')

        DownloadableDataset.__init__(self, 'http://ufldl.stanford.edu/housenumbers/', files_to_download, data_dir)
        self.loadDataset()

    def loadDataset(self):

        trainset = loadmat(self.data_dir + 'train_32x32.mat')
        testset = loadmat(self.data_dir + 'test_32x32.mat')

        if (self.with_extra):
            extraset = loadmat(self.data_dir + 'extra_32x32.mat')
            trainset = {'X': np.concatenate((trainset['X'], extraset['X']), axis = 3), 'y':np.concatenate((trainset['y'], extraset['y']))}

        trainset['X'] = np.rollaxis(trainset['X'], 3)
        testset['X'] = np.rollaxis(testset['X'], 3)

        num = len(trainset['y'])
        labels_info = np.ones(num)
        if self.leave_labels != -1:
            labels_info = np.concatenate(np.ones(self.leave_labels), np.zeros(num - self.leave_labels))

        val_split = int(num * self.val_split)

        self.train = DatasetBase(trainset['X'][val_split:], trainset['y'][val_split:], labels_info[val_split:])
        self.val = DatasetBase(trainset['X'][:val_split], trainset['y'][:val_split], labels_info[:val_split])
        self.test = DatasetBase(testset['X'], testset['y'])

        print('Train: inputs - ' + str(self.train.images.shape) + '\t outputs - ' + str(self.train.labels.shape))
        print('Val  : inputs - ' + str(self.val.images.shape) + '\t outputs - ' + str(self.val.labels.shape))
        print('Test : inputs - ' + str(self.test.images.shape) + '\t outputs - ' + str(self.test.labels.shape))


    def get_dataset_for_trainer(self, lazy_prepare = True, with_reconstruction = True):
        dataset = LazyPrepDataset((self.train, self.val, self.test), self.prep, no_shuffle = False)
        if not lazy_prepare:
            print('No lazy prepare!')
            self.train.images, self.train.labels = self.prep.preprocess(self.train.images, self.train.labels)
            self.val.images, self.val.labels = self.prep.preprocess(self.val.images, self.val.labels)
            self.test.images, self.test.labels = self.prep.preprocess(self.test.images, self.test.labels)
            #TODO: use decorators!
            base_dataset = SemiSupervisedDataset if self.leave_labels != -1 else ShuffleDataset
            dataset = base_dataset((self.train, self.val, self.test), no_shuffle = False)
        return DatasetWithReconstruction(dataset, with_reconstruction = with_reconstruction)

