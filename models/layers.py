import tensorflow as tf

def get_tf_shape_as_list(x):
    tf_shape = tf.shape(x)
    return tf.split(tf_shape, tf_shape.get_shape()[0])

def lstm_layer(x, hidden_size, seq_lengths, initial_state=None, seed=123123):
    cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer= \
                                   tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_lengths,
                      initial_state=initial_state, dtype=tf.float32)
    return outputs, final_state
    
def self_feeding_lstm_layer(hidden_size, max_output_sequence_length, initial_state, seed=123123):
    """
    An LSTM that will use the previous output as the current input
    """
    cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer= \
                                   tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    tf_batch_size = get_tf_shape_as_list(initial_state)[0]
    index = tf.constant(0)
    outputs = tf.zeros((0, tf_batch_size, hidden_size))
    next_input = tf.get_variable("self_feeding_lstm_initial_input", initializer=tf.zeros((hidden_size,)))
    next_state = initial_state
    def cond(i, outs, next_i, next_c):
        return tf.less(i, max_output_sequence_length)
    def body(i, outs, next_i, next_c):
        new_output, new_state = cell(next_i, next_c)
        return i + 1, tf.concat((os, new_output), axis=0), new_output, new_state 
    _, outputs, _, _ = tf.while_loop(cond, body, [index, outputs, next_input, next_state],
                  shape_invariants=None, parallel_iterations=1)
    return tf.transpose(outputs, (1, 0, 2))
    
def conv1d_transpose(x, output_shape, filters_out, window_size, stride, padding, seed=123123):
    """
    No bias, 1d version of tf.nn.conv2d_transpose
    """
    filter = tf.get_variable("conv1d_filter", shape=(1, window_size, filters_out, x.get_shape()[-1]),
                            dtype=tf.float32, 
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    return tf.nn.conv2d_transpose(
    x, filter, output_shape=output_shape,
    strides = (1, 1, stride, 1),
    padding=padding)
        
    