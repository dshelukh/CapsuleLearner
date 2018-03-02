'''
@author: Dmitry
'''

from CapsTools import *
import os
import urllib.request
import zipfile

class DatasetBase():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

class SizeInfo():
    def __init__(self, shape):
        self.shape = list(shape)

def sizes_from_tuple(t):
    l = list(t) if isinstance(t, tuple) else [t]
    sizes = []

    for i in l:
        sizes.append(SizeInfo(i.shape[1:]))
    return sizes

class Dataset():
    def __init__(self, base):
        self.train, self.val, self.test = base

    def get_shapes(self):
        X, y = self.get_batch(self.train, 0, 1)
        self.inputs_info = sizes_from_tuple(X)
        self.outputs_info = sizes_from_tuple(y)
        print ('Inputs:', [info.shape for info in self.inputs_info], 'outputs:', [info.shape for info in self.outputs_info])
        return self.inputs_info, self.outputs_info

    def get_dataset(self, dataset_name):
        return getattr(self, dataset_name)

    def get_num_batches(self, dataset, batch_size):
        num = len(dataset.images) // batch_size
        return num if len(dataset.images) % batch_size == 0 else num + 1

    def get_batch(self, dataset, num, batch_size):
        start, end = batch_size * num, batch_size * (num + 1)
        return dataset.images[start:end], dataset.labels[start:end]

class DownloadableDataset():
    def __init__(self, url, file_list, data_dir = 'data/', unzip_after_download = False):
        self.url = url
        self.file_list = file_list
        self.data_dir = data_dir
        self.unzip_after_download = unzip_after_download

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for file in file_list:
            self.download_if_needed(file)

    def download_if_needed(self, name):
        datafile = self.data_dir + name
        if not os.path.isfile(datafile):
            print('Downloading:', name)
            urllib.request.urlretrieve(self.url + name, datafile)
            print('Download complete!')
            if (self.unzip_after_download and name.endswith('.zip')):
                print('Unzipping file', datafile, 'to', self.data_dir)
                with zipfile.ZipFile(datafile, 'r') as zipped:
                    zipped.extractall(self.data_dir)
                print('Unzip complete!')

