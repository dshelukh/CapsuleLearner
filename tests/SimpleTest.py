'''
@author: Dmitry
'''
from common.dataset.SimpleDataset import SimpleDataset
from common.dataset.SimpleDataset import simple_dataset_num_labels
from common.network.NetworkBase import BasicNet
from common.elements.Elements import *
from Runners import ClassificationRunner
from CustomSaver import CustomSaver
from common.Trainer import Trainer, TrainerParams, Network


class SimpleTestNetworkConfig():
    def __init__(self):
        self.num_outputs = simple_dataset_num_labels
        self.with_reconstruction = False
        self.weight_decay = 0.0000
        self.convs = ConvLayout([
            CleverDenseElement(4, dropin = 0.05, decay = 0.99),
            ReshapeBlock([2, 2, 1]),
            #ConvData(4, (3, 3), 1, element = WeirdConvBlockElement),
            FinalizingElement(self.num_outputs, True, DenseElement)
            ], batch_norm = EmptyElement())

class SimpleTestNetwork(BasicNet):
    def __init__(self, config = SimpleTestNetworkConfig()):
        modes_dict = {
            'classification' : ClassificationRunner()
            }
        element_dict = {
            'encoder': ConvEncoder(config.convs, DenseElement), #CapsEncoder(),#ConvEncoder(),
            'predictor': EmptyElementConfig(),#DensePredict(),#CapsPredict(),#EmptyElementConfig(),
            }
        super().__init__(modes_dict, 'classification', element_dict, config = config)


dataset = SimpleDataset(num = 2)
save_folder = 'simple-test'
params = TrainerParams(0.001)
params.batch_size = 1
params.val_check_period = 0
params.max_epochs = 100
network_base = SimpleTestNetwork()
network = Network(network_base, *network_base.get_functions_for_trainer())

tf.reset_default_graph()

trainer = Trainer(network, dataset, params)

saver = CustomSaver(folders=[save_folder + '/classification', save_folder + '/classification/epoch'])
trainer.train(saver, restore_from_epochend = True)


