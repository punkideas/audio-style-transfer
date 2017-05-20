import tensorflow as tf
from layers import *

def seq2seq_ae(input_batch, seq_lengths, training, seed=12321):
    """
    input_batch: The (B, T, F) shaped tensor representing audio.  B - batch size, T - time dimension, F - features
    seq_lengths: (b,) shape tensor containing the sequence masks
    training: If true, use mean of batch for batch norm, otherwise using moving average
    """
    original_seq_lengths = seq_lengths
    p = "valid"
    net = batch_normalization(input_batch, training=training)
    layer1 = tf.layers.conv1d(net, 256, 11, strides=1, padding=p, use_bias=False,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    seq_lengths = tf.ceil((seq_lengths - 11 + 1) / 1.)
    net = batch_normalization(layer1, training=training)
    layer2 = tf.layers.conv1d(net, 256, 5, strides=2, padding=p, use_bias=False,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    seq_lengths = tf.ceil((seq_lengths - 5 + 1) / 2.)
    net = batch_normalization(layer2, training=training)
    layer3 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=False,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))   
    net = batch_normalization(layer3, training=training)
    seq_lengths = tf.ceil((seq_lengths - 3 + 1) / 2.)
    encoder_outputs, encoder_final_state = lstm_layer(net, 512, seq_lengths)
    decoder_outputs = self_feeding_lstm_layer(512, tf.reduce_max(seq_lengths), encoder_final_state)
    net = conv1d_transpose(decoder_outputs, tf.get_shape(layer2), 256, window_size=3, stride=2, padding=p)
    net = batch_normalization(net, training=training)
    net = conv1d_transpose(net, tf.get_shape(layer1), 256, window_size=5, stride=2, padding=p)
    net = batch_normalization(net, training=training)
    net = conv1d_transpose(net, tf.get_shape(input_batch), input_batch.get_shape()[-1], 
                            window_size=11, stride=1, padding=p)
    last_layer_bias = tf.get_variable("last_layer_bias", initilizer=tf.zeros((input_batch.get_shape()[-1])))
    net += last_layer_bias
    return net, [layer1, layer2, layer3, encoder_final_state]
    
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