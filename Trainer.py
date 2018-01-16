'''
@author: Dmitry
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from SimpleCapsNet import *
from CustomSaver import *
from tensorflow.examples.tutorials.mnist import input_data

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



class TrainerParams():
    def __init__(self, learning_rate = 0.001):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.max_epochs = 50
        self.batch_size = 500

        # early stopping params
        self.threshold = 10
        self.epsilon = 0.0001
        self.val_check_period = 30
        self.early_stopping = True

class Network():
    def __init__(self, network, loss, run = 'run', acc = None, minimizer = None):
        self.network = network
        self.loss = loss
        self.acc = acc
        self.runner = run
        self.minimizer = minimizer

    def run(self, *input_, training = True):
        func = getattr(self.network, self.runner)
        try:
            func(*input_, training = training)
        except TypeError as e:
            print('Looks like run function doesn\'t accept training argument', e)
            func(*input_)

    def loss_function(self, *targets):
        return np.array(getattr(self.network, self.loss)(*targets), ndmin = 1)

    def num_classified(self, *targets):
        if self.acc == None:
            return tf.constant(0)
        return getattr(self.network, self.acc)(*targets)

    def get_minimizers(self, optimizer, loss):
        if self.minimizer == None:
            return [optimizer.minimize(l) for l in loss]
        else:
            return getattr(self.network, self.minimizer)(optimizer, loss)

def placeholders_from_sizes(info_list):
    p = []
    for i in info_list:
        p.append(tf.placeholder(tf.float32, [None, *i.shape]))
    return tuple(p)

def print_losses(losses):
    if isinstance(losses, float):
        return '%.6f' % losses
    str = '[ '
    for loss in losses:
        str += ('%.6f ' % loss)
    str += ']'
    return str

class Trainer():

    def __init__(self, network, dataset, params = TrainerParams()):
        self.dataset = dataset
        self.params = params
        self.cur_epoch = 0
        self.cur_loss = [float('inf')]
        self.step_num = 0

        inputs_info, outputs_info = dataset.get_shapes()
        self.input_data = placeholders_from_sizes(inputs_info)
        self.targets = placeholders_from_sizes(outputs_info)
        self.training = tf.placeholder(tf.bool)

        self.network = network
        self.network.run(*self.input_data, training = self.training)
        self.loss = self.network.loss_function(*self.targets)
        self.acc = self.network.num_classified(*self.targets)
        self.minimizers = self.network.get_minimizers(self.params.optimizer, self.loss)

    def init_training(self, sess, saver, epochend = False):
        sess.run(tf.global_variables_initializer())
        result, data = saver.restore_session(sess, epochend)
        if result:
            if data:
                self.cur_epoch, self.cur_loss, self.step_num = int(data[0]), np.fromstring(data[1][1:-1], sep = ' '), int(data[2])
                print(self.cur_epoch, self.cur_loss, self.step_num)


    def run_on_dataset(self, sess, dataset, batch_size):
        total_acc = 0
        total_loss = np.zeros(len(self.loss))
        size = len(dataset.labels)
        num_batches = self.dataset.get_num_batches(dataset, batch_size)
        for i in range(0, num_batches):
            batch_images, batch_labels = self.dataset.get_batch(dataset, i, batch_size)
            cur_loss, cur_acc = sess.run((tuple(self.loss), self.acc), feed_dict={self.input_data: batch_images,
                                                                                  self.targets: batch_labels,
                                                                                  self.training: False})
            print (cur_acc)
            total_acc += cur_acc
            total_loss += np.array(cur_loss)
        return total_loss / size, total_acc / size

    def train(self, saver = None, augmentation = lambda *x: x, restore_from_epochend = False):
        dataset = self.dataset
        minimizers = self.minimizers
        
        if not saver:
            saver = CustomSaver()

        with tf.Session() as sess:
            self.init_training(sess, saver, restore_from_epochend)
            batch_size = self.params.batch_size
            waiting_for = 0
            train_dataset = dataset.get_dataset('train')
            num_batches = dataset.get_num_batches(train_dataset, batch_size)

            while(self.cur_epoch < self.params.max_epochs):
                self.cur_epoch += 1
                start_step = self.step_num % num_batches
                for i in range(start_step, num_batches):
                    batch_images, batch_labels = dataset.get_batch(train_dataset, i, batch_size)
                    inputs = sess.run(augmentation(*self.input_data), feed_dict={self.input_data: batch_images})
                    #writer = tf.summary.FileWriter("./tmp/log/smth.log", sess.graph)
                    losses = []
                    #for minimizer, loss in zip(minimizers, self.loss):
                    _, train_loss = sess.run((tuple(minimizers), tuple(self.loss)), feed_dict={self.input_data: inputs,
                                                                                           self.targets: batch_labels,
                                                                                           self.training: True})
                    #    losses.append(train_loss)
                    train_loss = np.array(train_loss)
                    #writer.close()
                    self.step_num += 1
                    print('Epoch', self.cur_epoch,', step', self.step_num, '. Training loss: ' + print_losses(train_loss / batch_size))

                    if (len(dataset.val.images) > 0 and self.step_num % self.params.val_check_period == 0):
                        new_loss, new_acc = self.run_on_dataset(sess, dataset.get_dataset('val'), batch_size)
                        if (sum(new_loss) < sum(self.cur_loss) - self.params.epsilon):
                            self.cur_loss = new_loss
                            save_path = saver.save_session(sess, params = (self.step_num, self.cur_epoch), save_data = (self.cur_epoch - 1, self.cur_loss, self.step_num))
                            print('Model saved: ', save_path, 'Validation loss:', print_losses(new_loss), 'Validation accuracy:', new_acc * 100)
                            waiting_for = 0
                        else:
                            waiting_for += 1
                            print('Model not saved, previous loss:', print_losses(self.cur_loss), ', new loss:', print_losses(new_loss), 'Accuracy: ', new_acc * 100)

                test_loss, test_acc = self.run_on_dataset(sess, dataset.get_dataset('test'), batch_size)
                print('Test accuracy after', self.cur_epoch, 'epoch: ', test_acc * 100, 'Test loss: ', print_losses(test_loss))
                saver.save_session(sess, True, (self.cur_epoch), save_data = (self.cur_epoch, test_loss, self.step_num))
                if (waiting_for > self.params.threshold and self.params.early_stopping):
                    print('Loss didn\'t improve for %d checks' % self.params.threshold)
                    break
        print('Training is completed!')

def augment_data(data, max_translate = (2, 2)):
    mtx, mty = max_translate
    N = tf.shape(data)[0]
    transform_mat = tf.concat([tf.ones([N, 1]), tf.zeros([N, 1]), tf.random_uniform([N, 1], minval = -mtx, maxval = mtx),
                               tf.zeros([N, 1]), tf.ones([N, 1]), tf.random_uniform([N, 1], minval = -mty, maxval = mty),
                               tf.zeros([N, 1]), tf.zeros([N, 1])], 1)
    return tf.contrib.image.transform(data, transform_mat)

if __name__ == "__main__":
    #from SvhnDataset import *
    #dataset = SvhnDataset(0.4, feature_range = (0, 1)).get_dataset_for_trainer(False)
    
    mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot=True, validation_size = 5000)
    dataset = Dataset(mnist)
    print(len(dataset.test.images),len(dataset.val.images),len(dataset.train.images))
    params = TrainerParams()
    params.val_check_period = 20

    network_base = SimpleCapsNet()
    network = Network(network_base, 'lossFunction', 'run', 'num_classified')
    trainer = Trainer(network, dataset, params)
    saver = CustomSaver()
    trainer.train(saver, augmentation = augment_data)
    print('Done!')
