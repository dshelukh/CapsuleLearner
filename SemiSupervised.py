'''
@author: Dmitry
'''
from Trainer import *
from SemiSupervisedNet import *
from SvhnDataset import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mode = 'semi'
save_folder = 'semi-supervised'
batch_size = 1000

need_resave = False
dataset = SvhnDataset(0.0, 1000).get_dataset_for_trainer()#AE_Dataset((train, val, test))
network_base = SemiSupervisedNetwork()
params = TrainerParams()
params.batch_size = batch_size

if (mode == 'ae' or mode == 'both'):
    tf.reset_default_graph()
    network = Network(network_base, 'ae_loss', 'run_autoencoder', 'num_classified_ae', minimizer = 'get_minimizers_ae')
    trainer = Trainer(network, dataset, params)
    saver = CustomSaver(folders=[save_folder + '/ae', save_folder + '/ae/epoch'])
    trainer.train(saver)
    need_resave = True

if (mode == 'semi' or mode == 'both'):
    tf.reset_default_graph()
    dataset.with_randoms = True
    network2 = Network(network_base, 'semi_loss', 'run_semi_supervised', 'num_classified_semi', 'get_minimizers_semi')
    trainer2 = Trainer(network2, dataset, params)
    saver2 = CustomSaver(folders=[save_folder + '/semi', save_folder + '/semi/epoch'])
    if (need_resave):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver2.restore_session(sess, filename = saver2.get_last_saved(save_folder + '/ae'))
            saver2.save_session(sess, False, (0, 0), (0, [float('inf'), float('inf')], 0))
            print('Resaved!')
    trainer2.train(saver2)

'''
tf.reset_default_graph()
trainer = Trainer(network2, dataset, params)
last_saved = tf.train.latest_checkpoint(base_folder + '/semi')
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
z = tf.placeholder(tf.float32, [None, 90])

with tf.Session() as sess:
    saver.restore_session(sess)
    (images, randoms), _ = dataset2.get_batch(dataset.val, 0, 6, False)
    
    #images2 = np.rollaxis(testset['X'], 3)[:6]
    #images2 = sess.run(network_base.img, feed_dict={trainer.input_data: images, trainer.training: False})
    images2 = sess.run(network_base.generated, feed_dict={trainer.input_data: (images, randoms), trainer.training: False})
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
'''