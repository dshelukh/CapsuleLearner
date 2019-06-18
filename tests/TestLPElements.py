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
from common.dataset.DatasetBase import DatasetwithCombinedInput
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
        self.print_train_output = False
        self.trainer = None

    def no_output(self, f):
        #sys.stdout = open(os.devnull, 'w') # hide messages
        with contextlib.redirect_stdout(None):
            f()
        #sys.stdout = sys.__stdout__ # restore output

    def control_output(self, f):
        if not self.print_train_output:
            self.no_output(f)
        else:
            f()

    def run_trainer(self, config = LpTestNetworkConfig()):
        network_base = LpTestNetwork(config = config)
        network = Network(network_base, *network_base.get_functions_for_trainer())

        self.trainer.resetTrainerWith(network, self.dataset, self.params)

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
        self.dataset = DatasetwithCombinedInput(SimpleDataset(num = 2, size = self.dataset_size))

        params = TrainerParams()
        params.batch_size = 10
        params.val_check_period = 0
        params.max_epochs = 1

        self.params = params

        tf.reset_default_graph()

    ### TEST CASES ###

    def test_step_called(self):
        config = self.get_config_with_mocked_element()
        self.control_output(lambda: self.run_trainer(config))

        self.assertGreaterEqual(config.convs.convs[0].step.call_count, 1, 'Step function should be called at least once')

    def test_step_executed_for_every_example(self):
        self.num_ex = tf.Variable(0, dtype = tf.int16, trainable = False)
        self.result = -1

        def step_mock_acc(X, y, training):
            num_ex_update = tf.assign(self.num_ex, self.num_ex + 1)
            with tf.control_dependencies([num_ex_update]):
                y = tf.identity(y) # using just return does not add update step to graph
                return y

        def save_num_ex():
            self.result = self.num_ex.eval()

        config = self.get_config_with_mocked_element(step_mock_acc)

        self.trainer.set_on_train_complete(save_num_ex)
        self.control_output(lambda: self.run_trainer(config))

        self.assertEqual(self.result, self.dataset_size * 2, 'Update should be executed for each training and testing example in dataset')

    def test_perfect_result(self):
        # to prove output of step can be compared to y
        self.test_acc = -1.0

        def save_acc(test_loss, test_acc):
            # test loss is pretty big, because y doesn't contain logits rather than actual probabilities
            self.test_acc = test_acc

        self.trainer.set_on_test_complete(save_acc)
        config = self.get_config_with_mocked_element()
        self.control_output(lambda: self.run_trainer(config))
        np.testing.assert_allclose(self.test_acc, [1.0], rtol = 0.0, atol = 1e-6, err_msg = 'Accuracy should be 1.0 as long as element returns y as its output')

    def test_creates_out_layer(self):
        config = LpTestNetworkConfig()
        self.control_output(lambda: self.run_trainer(config))

        layers = config.convs.convs[0].layers
        _, y = self.dataset.get_batch(self.dataset.get_dataset('train'), 0, 1)

        self.assertGreaterEqual(len(layers), 1, 'Should have at least one layer')
        self.assertEqual(len(layers[-1]), y.shape[-1], 'Number of elements in last layer should be equal to number of classes')

    def test_initializes_only_once(self):
        config = LpTestNetworkConfig()
        X, y = self.dataset.get_batch(self.dataset.get_dataset('train'), 0, 1)
        element = config.convs.convs[0]
        X = np.reshape(X[0], [-1, np.multiply.reduce(X[0].shape[1:])])

        element.init_out_layer(X.shape, y.shape)
        element.layers[-1][0].test_attr = 'testing'

        self.assertTrue(element.initialized, 'Initialized flag should be set to True after initialization')

        element.init_out_layer(X.shape, y.shape)
        has_test_attr = hasattr(element.layers[-1][0], 'test_attr')
        self.assertTrue(has_test_attr, 'New attribute should be still present')

