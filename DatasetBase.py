'''
@author: Dmitry
'''

from CapsTools import *
import os
import urllib.request
import zipfile
import tarfile

class DatasetBase():
    def __init__(self, images, labels, is_labeled = None):
        self.images = images
        self.labels = labels
        self.is_labeled = is_labeled

class SizeInfo():
    def __init__(self, shape):
        self.shape = list(shape)

def sizes_from_tuple(t):
    l = list(t) if isinstance(t, tuple) else [t]
    sizes = []

    for i in l:
        sizes.append(SizeInfo(i.shape[1:]))
    return sizes

class AbstractDataset():
    def get_shapes(self):
        pass

    def get_dataset(self, dataset_name):
        pass

    def get_size(self, dataset):
        pass

    def get_num_batches(self, dataset, batch_size):
        pass

    def get_batch(self, dataset, num, batch_size):
        raise(NotImplementedError( "Get batch is not implemented for dataset"))

class Dataset(AbstractDataset):
    def __init__(self, base):
        super()
        self.train, self.val, self.test = base

    def get_shapes(self):
        X, y = self.get_batch(self.train, 0, 1)
        self.inputs_info = sizes_from_tuple(X)
        self.outputs_info = sizes_from_tuple(y)
        print ('Inputs:', [info.shape for info in self.inputs_info], 'outputs:', [info.shape for info in self.outputs_info])
        return self.inputs_info, self.outputs_info

    def get_dataset(self, dataset_name):
        return getattr(self, dataset_name)

    def get_size(self, dataset):
        return len(dataset.images)

    def get_num_batches(self, dataset, batch_size):
        num = self.get_size(dataset) // batch_size
        return num if len(dataset.images) % batch_size == 0 else num + 1

    def get_batch_position(self, num, batch_size):
        return batch_size * num, batch_size * (num + 1)

    def get_batch(self, dataset, num, batch_size):
        start, end = self.get_batch_position(num, batch_size)
        return dataset.images[start:end], dataset.labels[start:end]

# Shuffle dataset each epoch if needed on first request (num == 0)
class ShuffleDataset(Dataset):
    def __init__(self, base, *args, no_shuffle = False):
        Dataset.__init__(self, base, *args)
        self.no_shuffle = no_shuffle
        self.shuffle_dict = {}

    def update_shuffle(self, dataset, shuffle):
        ind = np.arange(len(dataset.images))
        if (shuffle):
            np.random.shuffle(ind)
        self.shuffle_dict[dataset] = ind

    def get_batch(self, dataset, num, batch_size, shuffle = True):
        start, end = self.get_batch_position(num, batch_size)
        if (num == 0):
            self.update_shuffle(dataset, (shuffle and not self.no_shuffle))

        X = dataset.images[self.shuffle_dict[dataset][start:end]]
        y = dataset.labels[self.shuffle_dict[dataset][start:end]]
        return X, y

# Implements Dataset interface. Return initial images as output
class DatasetWithReconstruction(AbstractDataset):
    def __init__(self, dataset, with_reconstruction = True):
        super()
        self.dataset = dataset
        self.with_reconstruction = with_reconstruction

    def get_shapes(self):
        return self.dataset.get_shapes()

    def get_dataset(self, dataset_name):
        return self.dataset.get_dataset(dataset_name)

    def get_size(self, dataset):
        return self.dataset.get_size(dataset)

    def get_num_batches(self, dataset, batch_size):
        return self.dataset.get_num_batches(dataset, batch_size)

    def get_batch(self, dataset, num, batch_size, shuffle = False):
        X, y = self.dataset.get_batch(dataset, num, batch_size, shuffle)

        if self.with_reconstruction:
            y = (*y, X) if isinstance(y, tuple) else (y, X)
        return X, y

# Semi-supervised dataset: some images might have no labels + mechanism to guarantee sufficient number of labeled images per batch
class SemiSupervisedDataset(ShuffleDataset):
    def __init__(self, base, *args, code_generator = None, no_shuffle = True):
        ShuffleDataset.__init__(self, base, *args, no_shuffle = no_shuffle)
        self.code_generator = code_generator
        labels_info = base[0].is_labeled if base[0].is_labeled is not None else base[0].labels
        print('Number of training examples:', len(base[0].labels), 'labels left:', np.sum(labels_info))

    def set_code_generator(self, code_generator):
        self.code_generator = code_generator

    def get_batch(self, dataset, num, batch_size, shuffle = True, guaranteed_labels = 5):
        X, y = super().get_batch(dataset, num, batch_size, shuffle)
        labels = y[0] if (isinstance(y, tuple)) else y
        start, end = self.get_batch_position(num, batch_size)
        label_info = dataset.is_labeled if dataset.is_labeled is not None else np.sum(dataset.labels, axis = -1)

        cur_labels = label_info[self.shuffle_dict[dataset][start:end]]

        batch_size = np.minimum(batch_size, len(cur_labels))
        if guaranteed_labels > 0:
            guaranteed_labels = np.minimum(guaranteed_labels, batch_size)
            total_labels_left = int(np.sum(label_info))

            labels_needed = np.maximum(guaranteed_labels - int(np.sum(cur_labels)), 0)
            i = -1
            unlabeled = []
            while labels_needed > len(unlabeled):
                if np.sum(cur_labels[i]) == 0:
                    unlabeled.append(i)
                i -= 1

            # consider labeled data is in the beginning
            labeled_id = np.random.randint(total_labels_left, size = labels_needed)
            X[unlabeled] = dataset.images[labeled_id]
            labels[unlabeled] = dataset.labels[labeled_id]
            cur_labels[unlabeled] = label_info[labeled_id] #should be ones
            assert(np.sum(cur_labels) >= guaranteed_labels)

        if (self.code_generator):
            X = (X, self.code_generator.get_code(batch_size))

        return X, (labels, cur_labels)

#Dataset with lazy preprocessing
class LazyPrepDataset(ShuffleDataset):
    def __init__(self, base, preprocessor, *args, no_shuffle = True):
        ShuffleDataset.__init__(self, base, *args, no_shuffle = no_shuffle)
        self.prep = preprocessor

    def get_batch(self, dataset, num, batch_size, shuffle = True):
        X, y = super().get_batch(dataset, num, batch_size, shuffle)
        X, y = self.prep.preprocess(X, y)
        return X, y


class DownloadableDataset():
    def __init__(self, url, file_list, data_dir = 'data/', extract_after_download = False):
        self.url = url
        self.file_list = file_list
        self.data_dir = data_dir
        self.extract_after_download = extract_after_download

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for file in file_list:
            self.download_if_needed(file)

    def untar(self, datafile, gz = True):
        with tarfile.open(datafile, 'r:' + ('gz' if gz else '')) as arch:
            arch.extractall(self.data_dir)

    def unzip(self, datafile):
        with zipfile.ZipFile(datafile, 'r') as zipped:
            zipped.extractall(self.data_dir)

    def extract(self, name, datafile):
        print('Extracting contents of file', datafile, 'to', self.data_dir)

        if name.endswith('.zip'):
            self.unzip(datafile)
        elif name.endswith('.tar.gz') or name.endswith('.tar.gzip'):
            self.untar(datafile)
        elif name.endswith('.tar'):
            self.untar(datafile, False)

        print('Extraction complete!')

    def download_if_needed(self, name):
        datafile = self.data_dir + name
        if not os.path.isfile(datafile):
            print('Downloading:', name)
            urllib.request.urlretrieve(self.url + name, datafile)
            print('Download complete!')
            if (self.extract_after_download):
                self.extract(name, datafile)

