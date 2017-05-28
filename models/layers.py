import tensorflow as tf

def get_tf_shape_as_list(x, mix_python=True):
    tf_shape = tf.shape(x)
    tensor_shape_vals = tf.split(tf_shape, tf_shape.get_shape().as_list()[0])
    tf_shape_list = [tf.squeeze(_x) for _x in tensor_shape_vals]
    out = []
    for i, python_shape_val in enumerate(x.get_shape().as_list()):
        if mix_python and python_shape_val is not None:
            out.append(python_shape_val)
        else:
            out.append(tf_shape_list[i])
    return out

def lstm_layer(x, hidden_size, seq_lengths, initial_state=None, seed=123123):
    cell = tf.contrib.rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse, initializer= \
                                   tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_lengths,
                      initial_state=initial_state, dtype=tf.float32)
    return outputs, final_state

def unrolled_lstm_layer(x, hidden_size, seq_lengths, initial_state=None, seed=123123):
    cell = tf.contrib.rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse, initializer= \
                                   tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    B, T, C = get_tf_shape_as_list(x)
    assert type(T) == type(int(0))
    x = tf.split(x, T, axis=1)
    x = [tf.squeeze(ele, axis=1) for ele in x]
    cond = tf.reduce_all(tf.equal(seq_lengths, tf.ones_like(seq_lengths)*T))
    assert_op = tf.Assert(cond, [seq_lengths, T])
    with tf.control_dependencies([assert_op]):
        #outputs, final_state = tf.contrib.rnn.static_rnn(cell, x, sequence_length=seq_lengths,
        #              initial_state=initial_state, dtype=tf.float32)
        outputs = []
        if initial_state is None:
           initial_state = cell.zero_state(B, dtype=tf.float32)
        for t, next_i in enumerate(x):
            reuse = True if t > 0 else None
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                new_output, new_state = cell(next_i, initial_state)
            outputs.append(new_output)
            initial_state = new_state
        final_state = initial_state
    outputs = tf.concat([tf.expand_dims(o, axis=1) for o in outputs], axis=1)
    return outputs, final_state
    
def self_feeding_lstm_layer(max_output_sequence_length, initial_c_state, initial_m_state, seed=123123):
    """
    An LSTM that will use the previous output as the current input
    """
    tf_batch_size, hidden_size = get_tf_shape_as_list(initial_c_state)
    cell = tf.contrib.rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse, initializer= \
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
    shape_invariants=[index.get_shape(), tf.TensorShape([None, None, hidden_size]),
                      next_input.get_shape(), (initial_c_state.get_shape(), initial_m_state.get_shape())]
    _, outputs, _, _ = tf.while_loop(cond, body, [index, outputs, next_input, next_state],
                  shape_invariants=shape_invariants, parallel_iterations=1)
    return tf.transpose(outputs, (1, 0, 2))
    
def conv1d_transpose(x, output_shape, filters_out, window_size, stride, padding, use_bias = False, name=None, seed=123123):
    """
    output_shape must be a python list, and the last dimension must be known
    No bias, 1d version of tf.nn.conv2d_transpose
    """
    with tf.variable_scope(name):
        x = tf.expand_dims(x, axis=1)
        B, W, C = output_shape
        output_shape = tf.stack((B, 1, W, C))
        filter = tf.get_variable("conv1d_filter", shape=(1, window_size, filters_out, x.get_shape()[-1]),
                    dtype=tf.float32, 
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        x = tf.nn.conv2d_transpose(x, filter, output_shape=output_shape,
        strides = (1, 1, stride, 1), padding=padding)
        result = tf.squeeze(x, axis=[1])
        if use_bias:
            bias = tf.get_variable("conv1d_filter_bias", dtype=tf.float32, 
                                   initializer=tf.zeros(filters_out,))
            result += bias
        return result
        
  
def conv2d_transpose(x, filters_out, window_size, stride, padding, use_bias = False, name=None, seed=123123):
    """
    output_shape must be a python list, and the last dimension must be known
    No bias, 1d version of tf.nn.conv2d_transpose
    """
    assert padding == "SAME"
    output_shape = tf.stack([input_shape[0], stride*input_shape[1], stride*input_shape[2], output_dim])
    with tf.variable_scope(name):
        filter = tf.get_variable("conv2d_filter", shape=(window_size, window_size, \
                    filters_out, x.get_shape()[-1]), \
                    dtype=tf.float32, 
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
                    
        result = tf.nn.conv2d_transpose(x, filter, output_shape=output_shape,
        strides = (1, stride, stride, 1), padding=padding)
        
        if use_bias:
            bias = tf.get_variable("conv1d_filter_bias", dtype=tf.float32, 
                                   initializer=tf.zeros(filters_out,))
            result += bias
        return result
    

def bn(x, training, name=None):
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
	name=name,
	reuse=None)   

def leaky_relu(x, alpha = 0.01):
    return tf.maximum(x, alpha * x)
	
