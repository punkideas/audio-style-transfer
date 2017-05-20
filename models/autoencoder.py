import tensorflow as tf
from layers import *

def seq2seq_ae(input_batch, seq_lengths, is_training, seed=12321):
    """
    input_batch: The (B, T, F) shaped tensor representing audio.  B - batch size, T - time dimension, F - features
    seq_lengths: (b,) shape tensor containing the sequence masks
    training: If true, use mean of batch for batch norm, otherwise using moving average
    """
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

    encoder_outputs, encoder_final_state_tuple = lstm_layer(net, 512, seq_lengths)
    with tf.variable_scope("decoder"):
        decoder_outputs = self_feeding_lstm_layer(get_tf_shape_as_list(encoder_outputs)[1], encoder_final_state_tuple.c, encoder_final_state_tuple.h)

    # TODO double check that these conv1d_transposes are actually outputting the correct size    

    net = conv1d_transpose(decoder_outputs, get_tf_shape_as_list(layer2), 256, window_size=3, stride=2, padding=p, name="layer_d1")
    net = bn(net, is_training, "bn_d1")
    net = conv1d_transpose(net, get_tf_shape_as_list(layer1), 256, window_size=5, stride=2, padding=p, name="layer_d2")
    net = bn(net, is_training, "bn_d2")
    net = conv1d_transpose(net, get_tf_shape_as_list(input_batch), input_batch.get_shape()[-1], 
                            window_size=11, stride=1, padding=p, name="layer_d3")
    last_layer_bias = tf.get_variable("last_layer_bias", initializer=tf.zeros((input_batch.get_shape()[-1])))
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
    mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=2)
    loss = tf.nn.l2_loss((input_batch - ae_output) * mask) / tf.cast(get_tf_shape_as_list(input_batch)[0], tf.float32)
    return ae_output, layer_features, loss
