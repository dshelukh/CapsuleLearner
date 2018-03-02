'''
@author: Dmitry
'''

from EchoDataset import *
from SemiSupervisedNet import *

class EchoCancellationConfig():
    def __init__(self):
        self.lstm_size = 32
        self.num_layers = 2
        self.dropout = 0.2

        self.out_size = 1

class EchoCancellationNet(BasicNet):
    def __init__(self, config = EchoCancellationConfig()):
        modes_dict = {
            'ae' : AERunner(with_predictions = False, activation_to_use = (tf.sigmoid, tf.nn.sigmoid_cross_entropy_with_logits))
            }
        generator = LSTMGenerator()
        encoder = LSTMEncoder()
        element_dict = {
            'generator': generator,
            'encoder': encoder,
            'extractor': EmptyElementConfig(),
            }
        super(EchoCancellationNet, self).__init__(modes_dict, 'ae', element_dict, config = config)

import argparse

parser = argparse.ArgumentParser(description='Train semi-supervised net on Svhn')

mode_help = ('semi - run semi-supervised network only, ' +
             'ae - run autoencoder part only, ' +
             'both - run autoencoder and convert it to semi-supervised network')
parser.add_argument('--mode', default = 'ae', choices = ['ae', 'semi', 'both', 'none'],
                    help = mode_help)
parser.add_argument('-b', default = '16', type=int, help = 'batch size to use')
parser.add_argument('-l', default = '1000', type=int, help = 'number of labels to use in training')
parser.add_argument('--save', default = 'save', help = 'specify folder to save to')
args = vars(parser.parse_args())


mode = args['mode']
save_folder = args['save']
batch_size = args['b']
leave_num = args['l']

dataset = EchoDataset().get_dataset_for_trainer()
network_base = EchoCancellationNet()
network_base.set_mode('ae')
network = Network(network_base, *network_base.get_functions_for_trainer())
print(batch_size)
params = TrainerParams()
params.batch_size = batch_size
params.val_check_period = 20
params.optimizer = tf.train.AdamOptimizer(0.001)


if (mode == 'ae' or mode == 'both'):
    tf.reset_default_graph()
    trainer = Trainer(network, dataset, params)
    saver = CustomSaver(folders=[save_folder + '/ae', save_folder + '/ae/epoch'])
    trainer.train(saver, restore_from_epochend = False)

tf.reset_default_graph()
trainer = Trainer(network, dataset, params)
saver = CustomSaver(folders=[save_folder + '/ae', save_folder + '/ae/epoch'])
with tf.Session() as sess:
    saver.restore_session(sess, False)
    echoed, original = dataset.get_batch(dataset.test, 0, 1, False)
    result = sess.run(network_base.get_runner().img, feed_dict={trainer.input_data: (echoed), trainer.training: False})
    save_wav(echoed, './echoed.wav')
    save_wav(original, './original.wav')
    save_wav(result, './result.wav')
    print('Done!')


