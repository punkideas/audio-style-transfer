import tensorflow as tf
from layers import *

def seq2seq_ae(input_batch, seq_lengths, is_training, seed=12321):
    """
    input_batch: The (B, T, F) shaped tensor representing audio.  B - batch size, T - time dimension, F - features
    seq_lengths: (b,) shape tensor containing the sequence masks
    training: If true, use mean of batch for batch norm, otherwise using moving average
    """
    original_seq_lengths = seq_lengths
    p = "valid"
    net = bn(input_batch, is_training)
    layer1 = tf.layers.conv1d(net, 256, 11, strides=1, padding=p, use_bias=False,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 11 + 1) / 1.)
    net = bn(layer1, training=is_training)
    layer2 = tf.layers.conv1d(net, 256, 5, strides=2, padding=p, use_bias=False,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 5 + 1) / 2.)
    net = tf.layers.batch_normalization(layer2, training=is_training)
    layer3 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=False,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))   
    net = bn(layer3, training=is_training)
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 3 + 1) / 2.)
    encoder_outputs, encoder_final_state_tuple = lstm_layer(net, 512, seq_lengths)
    decoder_outputs = self_feeding_lstm_layer(tf.cast(tf.reduce_max(seq_lengths), tf.int32), encoder_final_state_tuple.c, encoder_final_state_tuple.h)
    net = conv1d_transpose(decoder_outputs, tf.get_shape(layer2), 256, window_size=3, stride=2, padding=p)
    net = bn(net, training=is_training)
    net = conv1d_transpose(net, tf.get_shape(layer1), 256, window_size=5, stride=2, padding=p)
    net = bn(net, training=is_training)
    net = conv1d_transpose(net, tf.get_shape(input_batch), input_batch.get_shape()[-1], 
                            window_size=11, stride=1, padding=p)
    last_layer_bias = tf.get_variable("last_layer_bias", initilizer=tf.zeros((input_batch.get_shape()[-1])))
    net += last_layer_bias
    return net, [layer1, layer2, layer3, encoder_final_state_tuple.c]
    
def seq2seq_ae_with_loss(input_batch, seq_lengths, training, seed=12321):
    """
    input_batch: The (B, T, F) shaped tensor representing audio.  B - batch size, T - time dimension, F - features
    seq_lengths: (b,) shape tensor containing the sequence masks
    training: If true, use mean of batch for batch norm, otherwise using moving average
    """
    ae_output, layer_features = seq2seq_ae(input_batch, seq_lengths, training, seed=seed)
    mask = tf.sequence_mask(seq_lengths, maxlen=input_batch.get_shape()[1]) # Fine if None, will take max length
    loss = tf.nn.l2_loss((input_batch - ae_output) * mask)
    return ae_output, layer_features, loss
