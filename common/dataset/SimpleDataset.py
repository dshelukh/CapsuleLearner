'''
@author: Dmitry
'''
import numpy as np
from common.dataset.DatasetBase import *

# A random square. Target value depends on corners
simple_dataset_num_labels = 16

class SimpleDataset(Dataset):
    def __init__(self, *args, num = 2, size = 20000):
        Dataset.__init__(self, ({}, {}, {}), *args)
        self.num = num
        self.num_labels = simple_dataset_num_labels
        self.size = size
        self.corner_mean = 0.9
        self.corner_std = 0.05
        self.other_mean = 0.3
        self.other_std = 0.15

    def onehot_y(self, data):
        one_hotter = np.eye(self.num_labels)
        return one_hotter[np.reshape(data - 1, [-1])]

    def get_batch(self, dataset, num, batch_size, *args, **kwargs):
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
        return self.size // batch_size if self.size % batch_size == 0 else self.size // batch_size + 1

    def get_dataset(self, name):
        return [None]

    def get_size(self, dataset):
        return self.size


