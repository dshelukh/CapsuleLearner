'''
@author: Dmitry
'''
import unittest
import numpy as np
import tensorflow as tf
import os
import sys
import contextlib

from common.dataset.SimpleDataset import SimpleDataset
from common.dataset.SimpleDataset import simple_dataset_num_labels
from common.network.NetworkBase import BasicNet
from common.elements.Elements import *
from common.elements.LinearProgrammingElement import *
from Runners import ClassificationRunner
from CustomSaver import CustomSaver
from common.Trainer import Trainer, TrainerParams, Network
from unittest.mock import Mock, DEFAULT


class LpTestNetworkConfig():
    def __init__(self):
        self.num_outputs = simple_dataset_num_labels
        self.with_reconstruction = False
        self.weight_decay = 0.0000
        self.convs = ConvLayout([
            LinearProgrammingDenseElement(self.num_outputs),
            ], batch_norm = EmptyElement())

class LpTestNetwork(BasicNet):
    def __init__(self, config = LpTestNetworkConfig()):
        modes_dict = {
            'classification' : ClassificationRunner()
            }
        element_dict = {
            'encoder': ConvEncoder(config.convs),
            'predictor': EmptyElementConfig(),
            }
        super().__init__(modes_dict, 'classification', element_dict, config = config)

class NoSaveSaver(CustomSaver):
    def __init__(self, allow_empty = False):
        super().__init__(allow_empty = allow_empty)

    def save_session(self, sess, epochend=False, params=None, save_data=None):
        return ''

class TestLPElements(unittest.TestCase):
    dataset_size = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = None

    def no_output(self, f):
        #sys.stdout = open(os.devnull, 'w') # hide messages
        with contextlib.redirect_stdout(None):
            f()
        #sys.stdout = sys.__stdout__ # restore output

    def run_trainer(self, network_base = LpTestNetwork()):
        dataset = SimpleDataset(num = 2, size = self.dataset_size)

        params = TrainerParams(0.001)
        params.batch_size = 10
        params.val_check_period = 0
        params.max_epochs = 1
        network = Network(network_base, *network_base.get_functions_for_trainer())

        self.trainer.resetTrainerWith(network, dataset, params)

        saver = NoSaveSaver(allow_empty = True)
        self.trainer.train(saver, restore_from_epochend = True)
        return

    def get_config_with_mocked_element(self, f = None):
        config = LpTestNetworkConfig()
        element = LinearProgrammingDenseElement(config.num_outputs)

        def step_mock(X, y, training):
            return y

        element.step = Mock(side_effect = step_mock if f is None else f)

        config.convs.convs = [element]
        return config



    ### TEST INITIALIZATION ###

    def setUp(self):
        self.trainer = Trainer(None, None)
        tf.reset_default_graph()

    ### TEST CASES ###

    def test_step_called(self):
        config = self.get_config_with_mocked_element()
        self.no_output(lambda: self.run_trainer(network_base = LpTestNetwork(config = config)))

        self.assertGreaterEqual(config.convs.convs[0].step.call_count, 1, 'Step function should be called at least once')

    def test_step_executed_for_every_example(self):
        self.acc = tf.Variable(0, dtype = tf.int16, trainable = False)
        self.result = -1

        def step_mock_acc(X, y, training):
            acc_update = tf.assign(self.acc, self.acc + 1)
            with tf.control_dependencies([acc_update]):
                y = tf.identity(y) # using just return does not add update step to the tree
                return y

        def save_acc():
            self.result = self.acc.eval()

        config = self.get_config_with_mocked_element(step_mock_acc)

        self.trainer.on_train_complete = save_acc
        self.no_output(lambda: self.run_trainer(network_base = LpTestNetwork(config = config)))

        self.assertEqual(self.result, self.dataset_size * 2, 'Update should be executed for each training and testing example in dataset')
