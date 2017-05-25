import tensorflow as tf
import os
import numpy as np
from glob import glob

def save(sess, saver, checkpoint_dir, experiment_name, step, tag=None):
        model_name = "hyperspectral_resnet.model"
        model_dir = "%s" % (experiment_name,)
        if tag is not None:
           model_dir = "%s_%s" % (model_dir, tag)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

def load(sess, saver, checkpoint_dir, experiment_name, tag=None):
        print(" [*] Reading checkpoints...")

        if checkpoint_dir is None:
           raise Exception("No checkpoint path, given, cannot load checkpoint")

        model_dir = "%s" % (experiment_name,)
        if tag is not None:
           model_dir = "%s_%s" % (model_dir, tag)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

def get_load_fns(checkpoint_dir, experiment_name, batch_size, tag=None):
        if checkpoint_dir is None:
           raise Exception("No checkpoint path, given, cannot load checkpoint")

        model_dir = "%s" % (experiment_name)
        if tag is not None:
           model_dir = "%s_%s" % (model_dir, tag)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        files = glob(os.path.join(checkpoint_dir, "hyperspectral_resnet.model*.meta"))
        print(files)
        out = []
        for _file in files:
            def load(sess, saver, _file=_file):
                model_path = _file[:-5]
                print("Loading")
                print(model_path)
                saver.restore(sess, model_path)
            out.append(load)
        return out

def print_number_of_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(variable.name)
        print(shape)
        print(len(shape))
        variable_parametes = 1
        for dim in shape:
            print(dim)
            variable_parametes *= dim.value
        print(variable_parametes)
        total_parameters += variable_parametes
    print("Total parameters: %d" % total_parameters)






