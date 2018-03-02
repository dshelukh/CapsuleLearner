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

# Like Dataset in Trainer but also returns initial images as outputs
# Can produce random noise vectors to use as an input
class DatasetWithReconstruction(Dataset):
    def __init__(self, base, *args, with_reconstruction = True, code_generator = None):
        self.code_generator = code_generator
        self.with_reconstruction = with_reconstruction
        self.num_labels = len(base[0].labels[0])
        Dataset.__init__(self, base, *args)
        print('num_labels:', self.num_labels, 'labels left:', np.sum(base[0].labels))

    def set_code_generator(self, code_generator):
        self.code_generator = code_generator

    def get_batch(self, dataset, num, batch_size, shuffle = True, guaranteed_labels = 5):
        start, end = batch_size * num, batch_size * (num + 1)
        ind = np.arange(len(dataset.images))
        if (shuffle):
            np.random.shuffle(ind)

        X = dataset.images[ind[start:end]]
        y = dataset.labels[ind[start:end]]

        batch_size = np.minimum(batch_size, len(y))
        if guaranteed_labels > 0:
            guaranteed_labels = np.minimum(guaranteed_labels, batch_size)
            total_labels_left = int(np.sum(dataset.labels))

            labels_needed = np.maximum(guaranteed_labels - int(np.sum(y)), 0)
            i = -1
            unlabeled = []
            while labels_needed > len(unlabeled):
                if np.sum(y[i]) == 0:
                    unlabeled.append(i)
                i -= 1

            # consider labeled data is in the beginning
            labeled_id = np.random.randint(total_labels_left, size = labels_needed)
            X[unlabeled] = dataset.images[labeled_id]
            y[unlabeled] = dataset.labels[labeled_id]
            assert(np.sum(y) >= guaranteed_labels)

        if (self.with_reconstruction):
            y = (y, X)

        if (self.code_generator):
            X = (X, self.code_generator.get_code(batch_size))

        return X, y

class SvhnDataset(DownloadableDataset):
    # leave_labels = -1 means use all labels
    def __init__(self, val_split = 0.3, leave_labels = -1, feature_range = (-1, 1), data_dir = 'data/'):
        self.num_labels = 10
        self.val_split = val_split
        self.leave_labels = leave_labels
        self.feature_range = feature_range
        DownloadableDataset.__init__(self, 'http://ufldl.stanford.edu/housenumbers/', ['train_32x32.mat', 'test_32x32.mat'], data_dir)
        self.loadDataset()

    def preprocess(self, images):
        images = np.rollaxis(images, 3)
        return scale(images, self.feature_range)

    def onehot_y(self, data):
        one_hotter = np.eye(self.num_labels)
        return one_hotter[np.reshape(data['y'] - 1, [-1])]

    def loadDataset(self):

        trainset = loadmat(self.data_dir + 'train_32x32.mat')
        testset = loadmat(self.data_dir + 'test_32x32.mat')

        train_labels = self.onehot_y(trainset)
        zero_labels = np.zeros_like(train_labels)
        l = len(train_labels)
        if self.leave_labels != -1:
            train_labels = np.concatenate((train_labels[:self.leave_labels], zero_labels[self.leave_labels:]), axis = 0)
            l = self.leave_labels
        train_imgs = self.preprocess(trainset['X'])

        val_split = int(l * self.val_split)

        self.train = DatasetBase(train_imgs[val_split:], train_labels[val_split:])
        self.val = DatasetBase(train_imgs[:val_split], train_labels[:val_split])
        self.test = DatasetBase(self.preprocess(testset['X']), self.onehot_y(testset))

        print('Train: inputs - ' + str(self.train.images.shape) + '\t outputs - ' + str(self.train.labels.shape))
        print('Val  : inputs - ' + str(self.val.images.shape) + '\t outputs - ' + str(self.val.labels.shape))
        print('Test : inputs - ' + str(self.test.images.shape) + '\t outputs - ' + str(self.test.labels.shape))


    def get_dataset_for_trainer(self, with_reconstruction = True):
        return DatasetWithReconstruction((self.train, self.val, self.test), with_reconstruction = with_reconstruction)



