import tensorflow as tf

def get_tf_shape_as_list(x):
    tf_shape = tf.shape(x)
    tensor_shape_vals = tf.split(tf_shape, tf_shape.get_shape().as_list()[0])
    return [tf.squeeze(x) for x in tensor_shape_vals]

def lstm_layer(x, hidden_size, seq_lengths, initial_state=None, seed=123123):
    cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer= \
                                   tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_lengths,
                      initial_state=initial_state, dtype=tf.float32)
    return outputs, final_state
    
def self_feeding_lstm_layer(max_output_sequence_length, initial_c_state, initial_m_state, seed=123123):
    """
    An LSTM that will use the previous output as the current input
    """
    tf_batch_size, hidden_size = get_tf_shape_as_list(initial_c_state)
    _, hidden_size = initial_c_state.get_shape().as_list()
    cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer= \
                                   tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    index = tf.constant(0)
    outputs = tf.zeros((0, tf_batch_size, hidden_size))
    next_input = tf.get_variable("self_feeding_lstm_initial_input", initializer=tf.zeros((1, hidden_size,)))
    next_input = tf.tile(next_input, (tf_batch_size, 1))
    next_state = (initial_c_state, initial_m_state)
    def cond(i, outs, next_i, next_c_m):
        return tf.less(i, max_output_sequence_length)
    def body(i, outs, next_i, next_c_m):
        new_output, new_state = cell(next_i, next_c_m)
        return i + 1, tf.concat((outs, tf.expand_dims(new_output, axis=0)), axis=0), new_output, (new_state.c, new_state.h) 
    _, outputs, _, _ = tf.while_loop(cond, body, [index, outputs, next_input, next_state],
                  shape_invariants=[index.get_shape(), tf.TensorShape([None, tf_batch_size, hidden_size]),
                                    next_input.get_shape(), (initial_c_state.get_shape(), initial_m_state.get_shape())],
                  parallel_iterations=1)
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
    

def bn(x, training):
    return tf.layers.batch_normalization(
	x,
	axis=-1,
	momentum=0.99,
	epsilon=0.001,
	center=True,
	scale=True,
	beta_initializer=tf.zeros_initializer(),
	gamma_initializer=tf.ones_initializer(),
	moving_mean_initializer=tf.zeros_initializer(),
	moving_variance_initializer=tf.ones_initializer(),
	beta_regularizer=None,
	gamma_regularizer=None,
	training=training,
	trainable=True,
	name=None,
	reuse=None)    
	
