import tensorflow as tf
import numpy as np
import random
import os
import pprint
import models.autoencoder as ae
import models.gan as gan

flags = tf.app.flags

flags.DEFINE_string('experiment_name', 'first_run', 'Name of the experiment being run')
flags.DEFINE_string('data_dir', None, "directory where data is stored")
flags.DEFINE_string('checkpoint_dir', './saved_checkpoints/', 'Checkpoint directory')
flags.DEFINE_integer('batch_size', 100, "The batch size")
flags.DEFINE_float('learning_rate', 1e-3, "The learning rate")
flags.DEFINE_float('gpu_usage', 0.96, "The gpu usage as a percentage")
flags.DEFINE_integer('num_epochs', 20, "The number of epochs to train")
flags.DEFINE_string('tag', None, "Optional tag, attached to checkpoints so runs with different tags have different checkpoints")
flags.DEFINE_string("best_model_tag", "max_val_acc_model", "The tag that identifies the directory which the best model is saved to (max_val_acc_model)")

FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    assert FLAGS.data_dir is not None
    FLAG_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
    FLAG_log_dir = os.path.join(FLAG_checkpoint_dir, "logs", FLAGS.experiment_name)

    FLAG_best_model_tag = FLAGS.best_model_tag
    if FLAGS.tag is not None:
        FLAG_best_model_tag = FLAGS.tag + "_" + FLAGS.best_model_tag
        
    suffix = "_ae"
    ae.train_seq2_seq_ae(FLAGS.data_dir, FLAGS.experiment_name + suffix, \
                FLAG_checkpoint_dir + suffix, FLAG_log_dir + suffix, FLAGS.batch_size, \
                FLAGS.learning_rate, FLAGS.num_epochs, FLAGS.gpu_usage, \
                FLAGS.tag, FLAG_best_model_tag)
     
    suffix = "_gan"
    gan.train_gan(FLAGS.data_dir, FLAGS.experiment_name + suffix, \
                FLAG_checkpoint_dir + suffix, FLAG_log_dir + suffix, FLAGS.batch_size, \
                FLAGS.learning_rate, FLAGS.num_epochs, FLAGS.gpu_usage, \
                FLAGS.tag, FLAG_best_model_tag)



if __name__ == '__main__':
    tf.app.run()







