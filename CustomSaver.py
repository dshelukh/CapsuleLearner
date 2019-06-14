'''
@author: Dmitry
'''
import tensorflow as tf
import os

class CustomSaver():
    epochend_filename_template = 'model_epochend%d'
    tmp_filename_template = "model%d_epoch%d"
    checkpoint_ext = '.ckpt'
    info_ext = '.info'

    def __init__(self, folders = ['tmp', 'tmp/epochend'], tmp_template = tmp_filename_template, epochend_template = epochend_filename_template, allow_empty = False):
        self.saver = tf.train.Saver(max_to_keep = 10, allow_empty = allow_empty)
        self.folders = self.get_folders(folders)
        for folder in self.folders:
            os.makedirs(folder, exist_ok = True)
        self.tmp_template = tmp_template
        self.epochend_template = epochend_template

    def get_folders(self, folders):
        if (len(folders) >= 1):
            return folders[:2]
        elif (len(folders == 1)):
            return [folders[0], folders[0]]
        else:
            return ['tmp', 'tmp']

    def get_folder(self, epochend):
        return self.folders[0] if not epochend else self.folders[1]

    def get_filename(self, epochend, params):
        return (self.tmp_template if not epochend else self.epochend_template) % params

    def save_session(self, sess, epochend = False, params = None, save_data = None):
        save_folder = self.get_folder(epochend)
        save_name = self.get_filename(epochend, params)
        file_name = '%s/%s' % (save_folder, save_name)

        self.saver.save(sess, file_name + CustomSaver.checkpoint_ext)
        if (save_data):
            with open(file_name + CustomSaver.info_ext, 'w') as file:
                file.write(';'.join(map(str, save_data)))
        return file_name + CustomSaver.checkpoint_ext

    # Thanks @RalphMao from https://github.com/tensorflow/tensorflow/issues/312
    def optimistic_restore(self, session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        # One more saver here to restore only
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    def restore_session(self, sess, epochend = False, filename = None):
        if not filename:
            folder = self.get_folder(epochend)
            filename = self.get_last_saved(folder)
        data = None
        if (filename):
            print('Restoring session from', filename)
            self.optimistic_restore(sess, filename)
            info_file = os.path.splitext(filename)[0] + CustomSaver.info_ext
            if (os.path.isfile(info_file)):
                with open(info_file, 'r') as file:
                    lines = ';'.join(file.readlines())
                    data = lines.split(";")
            return True, data
        else:
            print('No checkpoint found in folder', folder)
            return False, data

    def get_last_saved(self, folder):
        return tf.train.latest_checkpoint(folder)


