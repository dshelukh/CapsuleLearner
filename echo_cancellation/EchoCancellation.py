'''
@author: Dmitry
'''

from EchoDataset import *
from SemiSupervisedNet import *

class EchoCancellationConfig():
    def __init__(self):
        self.convs = [
                ConvData(16, (7, 1), (1, 1), activation = tf.tanh),
                ConvData(16, (7, 1), (1, 1), activation = tf.tanh)
            ]
        self.lstm1_size = 64
        self.lstm1_layers = 1
        self.dense_size = 16
        self.lstm2_size = 64
        self.lstm2_layers = 1
        self.dropout = 0.0

        self.out_size = 32
        self.chunk_size = 32

class EchoCancellationNet(BasicNet):
    def __init__(self, config = EchoCancellationConfig()):
        modes_dict = {
            'ae' : AERunner(with_predictions = False) #, activation_to_use = (tf.sigmoid, tf.nn.sigmoid_cross_entropy_with_logits))
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

parser = argparse.ArgumentParser(description='Train echo cancellation net')

mode_help = ('ae - train autoencoder for echo cancellation, ' +
             'none - run trained network (produces original, echoed and result files in save folder)')
parser.add_argument('--mode', default = 'ae', choices = ['ae', 'none'],
                    help = mode_help)
parser.add_argument('-b', default = '16', type=int, help = 'batch size to use')
parser.add_argument('--save', default = 'save', help = 'specify folder to save to')
args = vars(parser.parse_args())


mode = args['mode']
save_folder = args['save']
batch_size = args['b']

config = EchoCancellationConfig()
dataset = EchoDataset().get_dataset_for_trainer(step_size = config.chunk_size)
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
echo_levels = [0.2, 0.4, 0.65]
echo_delays = [0.2, 0.5, 0.7]
with tf.Session() as sess:
    saver.restore_session(sess, False)
    echoed, original = dataset.get_batch(dataset.test, 0, 1, False)
    result = sess.run(network_base.get_runner().img, feed_dict={trainer.input_data: (echoed), trainer.training: False})

    save_wav(np.copy(original), save_folder + '/original.wav')
    original = np.reshape(original, [-1])
    input_shape = echoed.shape
    for i, level in enumerate(echo_levels):
        for j, delay in enumerate(echo_delays):
            echoed, _ = add_echo(np.copy(original), delay, level)
            result = sess.run(network_base.get_runner().img, feed_dict={trainer.input_data: (np.reshape(echoed, input_shape)), trainer.training: False})
            save_wav(echoed, save_folder + '/echoed_l%d_d%d.wav' % (i, j))
            save_wav(result, save_folder + '/result_l%d_d%d.wav' % (i, j))

    print('Done!')


