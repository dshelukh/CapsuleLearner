'''
@author: Dmitry
'''
from Trainer import *
from SemiSupervisedNet import *
from SvhnDataset import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Train semi-supervised net on Svhn')

mode_help = ('semi - run semi-supervised network only, ' +
             'ae - run autoencoder part only, ' +
             'both - run autoencoder and convert it to semi-supervised network')
parser.add_argument('--mode', default = 'semi', choices = ['ae', 'semi', 'both', 'none'],
                    help = mode_help)
parser.add_argument('-b', default = '150', type=int, help = 'batch size to use')
parser.add_argument('-l', default = '1000', type=int, help = 'number of labels to use in training')
parser.add_argument('--save', default = 'semi-supervised', help = 'specify folder to save to')
args = vars(parser.parse_args())


mode = args['mode']
save_folder = args['save']
batch_size = args['b']
leave_num = args['l']

need_resave = False
dataset = SvhnDataset(0.3, leave_num).get_dataset_for_trainer()
network_base = SemiCapsNet() #SemiSupCapsNet() #SemiSupervisedNetwork()

params = TrainerParams()
params.batch_size = batch_size
params.val_check_period = 50
params.optimizer = tf.train.AdamOptimizer(0.001)


if (mode == 'ae' or mode == 'both'):
    tf.reset_default_graph()
    network_base.set_mode('ae')
    network = Network(network_base, *network_base.get_functions_for_trainer())
    trainer = Trainer(network, dataset, params)
    saver = CustomSaver(folders=[save_folder + '/ae', save_folder + '/ae/epoch'])
    trainer.train(saver)
    need_resave = True

network2 = Network(network_base, *network_base.get_functions_for_trainer())
if (mode == 'semi' or mode == 'both'):
    params.early_stopping = False
    tf.reset_default_graph()
    dataset.set_code_generator(network_base.get_code_generator())
    (images, randoms), _ = dataset.get_batch(dataset.val, 0, batch_size, False)
    network_base.set_reference_batch(randoms)
    trainer2 = Trainer(network2, dataset, params)
    saver2 = CustomSaver(folders=[save_folder + '/semi', save_folder + '/semi/epoch'])
    if (need_resave):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver2.restore_session(sess, filename = saver2.get_last_saved(save_folder + '/ae'))
            saver2.save_session(sess, False, (0, 0), (0, [float('inf'), float('inf')], 0))
            print('Resaved!')
    trainer2.train(saver2, restore_from_epochend = not need_resave)


tf.reset_default_graph()
dataset.set_code_generator(network_base.get_code_generator())
trainer = Trainer(network2, dataset, params)
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
z = tf.placeholder(tf.float32, [None, network_base.config.code_size])
saver = CustomSaver([save_folder + '/semi', save_folder + '/semi/epoch'])
with tf.Session() as sess:
    saver.restore_session(sess, True)
    (images, randoms), _ = dataset.get_batch(dataset.val, 0, 6, False)
    
    #images2 = np.rollaxis(testset['X'], 3)[:6]
    #images2 = sess.run(network_base.img, feed_dict={trainer.input_data: images, trainer.training: False})
    images2 = sess.run(tf.tanh(network_base.get_runner().generated), feed_dict={trainer.input_data: (images, randoms), trainer.training: False})
    images = np.concatenate((images, images2), axis = 0)
    images = unscale(images)
    fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12,3),)
    for ii, ax in zip(images, axes.flatten()):
        ax.axis('off')
        
        ax.set_adjustable('box-forced')
        ax.imshow(ii)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
