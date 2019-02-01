'''
@author: Dmitry
'''
from common.dataset.DatasetBase import *

import pickle as pkl
import tarfile
import os.path
import numpy as np
'''
import time

import os
import urllib.request

from scipy.io import loadmat
import sys
'''
def random_translate(image, vbound = (-2, 2), hbound = (-2, 2)):
    v = np.random.randint(vbound[0], vbound[1] + 1)
    h = np.random.randint(hbound[0], hbound[1] + 1)
    image = np.roll(image, (v, h), axis = (0, 1))
    return image

def get_translator(vbound, hbound):
    return lambda x: random_translate(x, vbound, hbound)

#Preprocess from .mat files
class CifarPreprocessor():
    def __init__(self, num_labels, feature_range = (-1, 1)):
        self.num_labels = num_labels
        self.feature_range = feature_range

    def preprocess_X(self, images, augmentation):
        retVal = scale(images, self.feature_range).astype(np.float32)
        retVal = np.reshape(retVal, [-1, 3, 32, 32])
        # from nchw to nhwc
        retVal = np.transpose(retVal, [0, 2, 3, 1])
        if augmentation:
            #retVal = np.apply_along_axis(get_translator((-4, 4), (-4, 4)), 0, retVal)
            func = np.vectorize(get_translator((-4, 4), (-4, 4)), signature = '(n,m,k)->(n,m,k)')
            retVal = func(retVal)
        return retVal

    def onehot_y(self, data):
        one_hotter = np.eye(self.num_labels)
        return one_hotter[np.reshape(data - 1, [-1])]

    def preprocess(self, imgs, labels, augmentation = True):
        if imgs.size > 0 and labels.size > 0:
            labels = self.onehot_y(labels)
            imgs = self.preprocess_X(imgs, augmentation)
        return imgs, labels

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pkl.load(fo, encoding='bytes')
    return dict

class Cifar10Dataset(DownloadableDataset):
    # leave_labels = -1 means use all labels

    def __init__(self, val_split = 0.0, leave_labels = -1, feature_range = (-1, 1), data_dir = 'data-cifar10/'):

        files_to_download = ['cifar-10-python.tar.gz']
        DownloadableDataset.__init__(self, 'https://www.cs.toronto.edu/~kriz/', files_to_download, data_dir)

        self.val_split = val_split
        self.leave_labels = leave_labels
        self.prep = CifarPreprocessor(10, feature_range = feature_range)

        self.cifar_batches_dir = self.data_dir + 'cifar-10-batches-py/'
        self.cifar_files_train = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.cifar_files_test = ['test_batch']
        self.cifar_files_all = self.cifar_files_train + self.cifar_files_test
        self.cifar_pic_size = 32 * 32 * 3
        self.cifar_data_tag = b'data'
        self.cifar_labels_tag = b'labels'
        self.loadDataset()

    def check_files_exist(self):
        for f in self.cifar_files_all:
            if not os.path.isfile(self.cifar_batches_dir + f):
                return False
        return True

    def unpickle_all(self, files):
        data = np.array([], dtype = np.uint8).reshape([0, self.cifar_pic_size])
        labels = np.array([], dtype = np.uint8).reshape([0])
        for f in files:
            dict = unpickle(self.cifar_batches_dir + f)
            data = np.concatenate([data, np.array(dict[self.cifar_data_tag])], axis = 0)
            labels = np.concatenate([labels, np.array(dict[self.cifar_labels_tag])], axis = 0)
        return data, labels

    def loadDataset(self):
        # untar if needed
        exist = self.check_files_exist()
        if not exist:
            for f in self.file_list:
                self.extract(f, self.data_dir + f)

        # unpickle
        self.train_data, self.train_labels = self.unpickle_all(self.cifar_files_train)
        self.test_data, self.test_labels = self.unpickle_all(self.cifar_files_test)

        num = len(self.train_data)
        labels_info = np.ones(num)
        if self.leave_labels != -1:
            labels_info = np.concatenate(np.ones(self.leave_labels), np.zeros(num - self.leave_labels))

        val_split = int(num * self.val_split)

        self.train = DatasetBase(self.train_data[val_split:], self.train_labels[val_split:], labels_info[val_split:])
        self.val = DatasetBase(self.train_data[:val_split], self.train_labels[:val_split], labels_info[:val_split])
        self.test = DatasetBase(self.test_data, self.test_labels)

        print('Train: inputs - ' + str(self.train.images.shape) + '\t outputs - ' + str(self.train.labels.shape))
        print('Val  : inputs - ' + str(self.val.images.shape) + '\t outputs - ' + str(self.val.labels.shape))
        print('Test : inputs - ' + str(self.test.images.shape) + '\t outputs - ' + str(self.test.labels.shape))
        '''
        trainset = loadmat(self.data_dir + 'train_32x32.mat')
        testset = loadmat(self.data_dir + 'test_32x32.mat')

        if (self.with_extra):
            extraset = loadmat(self.data_dir + 'extra_32x32.mat')
            trainset = {'X': np.concatenate((trainset['X'], extraset['X']), axis = 3), 'y':np.concatenate((trainset['y'], extraset['y']))}

        trainset['X'] = np.rollaxis(trainset['X'], 3)
        testset['X'] = np.rollaxis(testset['X'], 3)

        

        
        '''

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

if __name__ == "__main__":
    dataset = Cifar10Dataset()
    print (dataset.data[b'labels'])
    
