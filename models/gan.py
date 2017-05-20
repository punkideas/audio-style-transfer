import tensorflow as tf
from layers import *

def generator(sample_noise):
    pass

def discriminator(d_in, seq_lengths, is_training, seed=2376):
    original_seq_lengths = seq_lengths
    p = "VALID"
    net = bn(input_batch, is_training, "bn1")
    layer1 = tf.layers.conv1d(net, 256, 11, strides=1, padding=p, use_bias=False, name="layer1",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 11 + 1) / 1.)
    net = bn(layer1, is_training, "bn2")
    layer2 = tf.layers.conv1d(net, 256, 5, strides=2, padding=p, use_bias=False, name="layer2",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 5 + 1) / 2.)
    net = bn(layer2, is_training, "bn3")
    layer3 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=False, name="layer3",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    net = bn(layer3, is_training, "bn4")
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 3 + 1) / 2.)
    layer4 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=False, name="layer4",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    net = bn(layer3, is_training, "bn5")
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 3 + 1) / 2.)

    encoder_outputs, encoder_final_state_tuple = lstm_layer(net, 512, seq_lengths)
    net = tf.layers.dense(encoder_final_state_tuple.h, 1)
    return net, [layer1, layer2, layer3, layer4, encoder_final_state_tuple.c]

def wgan_loss():
    # https://arxiv.org/abs/1704.00028
    pass # TODO


