'''
@author: Dmitry
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from SimpleCapsNet import *
from CustomSaver import *
from common.dataset.DatasetBase import Dataset


class EpochScheduler():
    def __init__(self, start, multiplicator, epoch_step):
        self.start = start
        self.multiplicator = multiplicator
        self.epoch_step = epoch_step
        self.cur = None

    def get(self, epoch, *args):
        if (self.cur is None):
            self.cur = self.start
            for i in range(epoch // self.epoch_step):
                self.cur *= self.multiplicator
        else:
            if (epoch % self.epoch_step == 0):
                self.cur *= self.multiplicator
        return self.cur

class EpochListScheduler():
    def __init__(self, start, pairs):
        self.start = start
        self.pairs = pairs
        self.cur = None
        self.cur_num = 0

    def get(self, epoch, *args):
        if (self.cur is None):
            self.cur = self.start
        for num, val in self.pairs:
            if (num <= epoch and val <= self.cur):
                self.cur = val	    
        return self.cur

class TrainerParams():
    def __init__(self, learning_rate = 0.001):
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = learning_rate if(isinstance(learning_rate, EpochScheduler)) else EpochScheduler(learning_rate, 1.0, 100)
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
        self.network = network

        if (dataset is not None) and (network is not None):
            self.resetTrainer()
        else:
            self.ready = False

    def resetTrainer(self):
        self.cur_epoch = 0
        self.cur_loss = [float('inf')]
        self.step_num = 0

        self.setIOPlaceholders()
        self.setTrainingVariables()

        self.run_network()

        self.trainer_output = True
        self.ready = True

    def resetTrainerWith(self, network, dataset, params):
        self.dataset = dataset
        self.params = params
        self.network = network
        self.resetTrainer()

    # Create placeholder variables to match sizes of input and output. First dimension(batch size) should be None
    def setIOPlaceholders(self):
        inputs_info, outputs_info = self.dataset.get_shapes()
        self.input_data = placeholders_from_sizes(inputs_info)
        self.targets = placeholders_from_sizes(outputs_info)

    # Placeholders for variables used during training. Currently it's training(bool) and learning_rate(float32)
    def setTrainingVariables(self):
        self.training = tf.placeholder(tf.bool)
        self.learning_rate = tf.placeholder(tf.float32)

    # execute network functions to obtain loss, accuracy and minimizers tensors
    def run_network(self):
        self.network.run(self.input_data, training = self.training)
        self.loss = self.network.loss_function(*self.targets)
        self.acc = self.network.num_classified(*self.targets)
        optimizer = self.params.optimizer if isinstance(self.params.optimizer, tf.train.Optimizer) else self.params.optimizer(self.learning_rate)
        self.minimizers = self.network.get_minimizers(optimizer, self.loss)

    # executes after restoring session from file
    def on_data_load(self):
        if self.trainer_output:
            print('Training state loaded:', self.cur_epoch, print_losses(self.cur_loss), self.step_num)

    def set_on_data_load(self, on_data_load):
        self.on_data_load = on_data_load

    # executes after finishing training
    def on_train_complete(self):
        print('Training is completed!')

    def set_on_train_complete(self, on_train_complete):
        self.on_train_complete = on_train_complete

    # executes after every training batch
    def on_batch(self, loss):
        print('Epoch', self.cur_epoch,', step', self.step_num, '. Training loss: ' + print_losses(loss))

    def set_on_batch(self, on_batch):
        self.on_batch = on_batch

    # executes after tests are run
    def on_test_complete(self, test_loss, test_acc):
        print('Test accuracy after', self.cur_epoch, 'epoch: ', test_acc * 100, 'Test loss: ', print_losses(test_loss))

    def set_on_test_complete(self, on_test_complete):
        self.on_test_complete = on_test_complete

    # initialize training or restore session from file
    def init_training(self, sess, saver, epochend = False):
        sess.run(tf.global_variables_initializer())
        result, data = saver.restore_session(sess, epochend)
        if result:
            if data:
                self.cur_epoch, self.cur_loss, self.step_num = int(data[0]), np.fromstring(data[1][1:-1], sep = ' '), int(data[2])
                self.on_data_load()


    # calculate total loss and total accuracy on dataset
    def run_on_dataset(self, sess, dataset, batch_size):
        total_acc = 0
        total_loss = np.zeros(len(self.loss))
        size = self.dataset.get_size(dataset)

        if size == 0:
            return 0.0, 0.0

        num_batches = self.dataset.get_num_batches(dataset, batch_size)
        for i in range(0, num_batches):
            batch_images, batch_labels = self.dataset.get_batch(dataset, i, batch_size, training = False)
            cur_loss, cur_acc = sess.run((tuple(self.loss), self.acc), feed_dict={self.input_data: batch_images,
                                                                                  self.targets: batch_labels,
                                                                                  self.training: False})
            print (cur_acc)
            total_acc += cur_acc
            total_loss += np.array(cur_loss)
        return total_loss / size, total_acc / size

    def update_params(self, epoch, step):
        self.learning_rate_value = self.params.learning_rate.get(epoch, step)
        print('Learning rate value: %.6f ' % self.learning_rate_value)

    def train(self, saver = None, augmentation = lambda *x: x, restore_from_epochend = False):
        if not self.ready:
            self.resetTrainer()

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
                self.update_params(self.cur_epoch, self.step_num)
                self.cur_epoch += 1
                start_step = self.step_num % num_batches
                for i in range(start_step, num_batches):
                    batch_images, batch_labels = dataset.get_batch(train_dataset, i, batch_size, True)
                    #writer = tf.summary.FileWriter("./tmp/log/smth.log", sess.graph)
                    losses = []
                    _, train_loss = sess.run((tuple(minimizers), tuple(self.loss)), feed_dict={self.input_data: batch_images,
                                                                                           self.targets: batch_labels,
                                                                                           self.training: True,
                                                                                           self.learning_rate: self.learning_rate_value})
                    train_loss = np.array(train_loss)
                    #writer.close()
                    self.step_num += 1
                    self.on_batch(train_loss / batch_size)

                    if (self.params.val_check_period > 0 and self.step_num % self.params.val_check_period == 0):
                        if (dataset.get_dataset('val') and len(dataset.get_dataset('val').images) > 0):
                            new_loss, new_acc = self.run_on_dataset(sess, dataset.get_dataset('val'), batch_size)
                            if (sum(new_loss) < sum(self.cur_loss) - self.params.epsilon):
                                self.cur_loss = new_loss
                                save_path = saver.save_session(sess, params = (self.step_num, self.cur_epoch), save_data = (self.cur_epoch - 1, print_losses(self.cur_loss), self.step_num))
                                print('Model saved: ', save_path, 'Validation loss:', print_losses(new_loss), 'Validation accuracy:', new_acc * 100)
                                waiting_for = 0
                            else:
                                waiting_for += 1
                                print('Model not saved, previous loss:', print_losses(self.cur_loss), ', new loss:', print_losses(new_loss), 'Accuracy: ', new_acc * 100)
                        else: # No validation dataset but val check is set - just save
                            save_path = saver.save_session(sess, params = (self.step_num, self.cur_epoch), save_data = (self.cur_epoch - 1, print_losses(self.cur_loss), self.step_num))
                            print('Model saved:', save_path)

                test_dataset = dataset.get_dataset('test')
                if (test_dataset and dataset.get_size(test_dataset) > 0):
                    test_loss, test_acc = self.run_on_dataset(sess, dataset.get_dataset('test'), batch_size)
                    self.on_test_complete(test_loss, test_acc)
                    saver.save_session(sess, True, (self.cur_epoch), save_data = (self.cur_epoch, print_losses(test_loss), self.step_num))
                if (waiting_for > self.params.threshold and self.params.early_stopping):
                    print('Loss didn\'t improve for %d checks' % self.params.threshold)
                    break
            self.on_train_complete()
        self.ready = False

def augment_data(data, max_translate = (2, 2)):
    mtx, mty = max_translate
    N = tf.shape(data)[0]
    transform_mat = tf.concat([tf.ones([N, 1]), tf.zeros([N, 1]), tf.random_uniform([N, 1], minval = -mtx, maxval = mtx),
                               tf.zeros([N, 1]), tf.ones([N, 1]), tf.random_uniform([N, 1], minval = -mty, maxval = mty),
                               tf.zeros([N, 1]), tf.zeros([N, 1])], 1)
    return tf.contrib.image.transform(data, transform_mat)

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
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
