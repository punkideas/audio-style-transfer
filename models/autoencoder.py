import tensorflow as tf
from layers import *
from featurization.featurization import read_data_dir
from utils import *

def seq2seq_ae(input_batch, seq_lengths, is_training, seed=12321):
    """
    input_batch: The (B, T, F) shaped tensor representing audio.  B - batch size, T - time dimension, F - features
    seq_lengths: (b,) shape tensor containing the sequence masks
    training: If true, use mean of batch for batch norm, otherwise using moving average
    """
    original_seq_lengths = seq_lengths
    p = "VALID"
    net = bn(input_batch, is_training, "bn1")
    
    layer1 = tf.layers.conv1d(net, 256, 11, strides=1, padding=p, use_bias=True, name="layer1",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    layer1 = tf.nn.relu(layer1)
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 11 + 1) / 1.)
    net = bn(layer1, is_training, "bn2")

    layer2 = tf.layers.conv1d(net, 256, 5, strides=2, padding=p, use_bias=True, name="layer2",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    layer2 = tf.nn.relu(layer2)
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 5 + 1) / 2.)
    net = bn(layer2, is_training, "bn3")

    layer3 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=True, name="layer3",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))   
    layer3 = tf.nn.relu(layer3)
    seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 3 + 1) / 2.)
    net = bn(layer3, is_training, "bn4")

    encoder_outputs, encoder_final_state_tuple = lstm_layer(net, 512, seq_lengths)
    with tf.variable_scope("decoder"):
        decoder_outputs = self_feeding_lstm_layer(get_tf_shape_as_list(encoder_outputs)[1], encoder_final_state_tuple.c, encoder_final_state_tuple.h)

    # TODO double check that these conv1d_transposes are actually outputting the correct size    

    net = conv1d_transpose(decoder_outputs, get_tf_shape_as_list(layer2), 256, window_size=3, 
                            stride=2, padding=p, use_bias=True, name="layer_d1")
    net = tf.nn.relu(net)
    net = bn(net, is_training, "bn_d1")

    net = conv1d_transpose(net, get_tf_shape_as_list(layer1), 256, window_size=5, stride=2, 
                            padding=p, use_bias=True, name="layer_d2")
    net = tf.nn.relu(net)
    net = bn(net, is_training, "bn_d2")

    net = conv1d_transpose(net, get_tf_shape_as_list(input_batch)[:-1] + [256], 256, 
                            window_size=11, stride=1, padding=p, use_bias=True, name="layer_d3")
    net = tf.nn.relu(net)
    net = bn(net, is_training, "bn_d3")

    net = tf.layers.conv1d(net, input_batch.get_shape()[-1], 1, strides=1, padding="VALID", 
                        use_bias=True, name="final_layer",
                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    # TODO final activation function depends on data normalization
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

    
def setup_seq2seq_ae(inputs, seq_lengths, learning_rate=1e-3, is_training=True, global_step=None):
    # You can feed a placeholder for learning_rate if you want to use decay
    B, T, C = get_tf_shape_as_list(inputs)
    
    ae_output, layer_features, loss = seq2seq_ae_with_loss(inputs, seq_lengths, is_training)
    
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = solver.minimize(loss, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    all_ops = update_ops + [train_step]
    train_step = tf.group(*all_ops)

    def run_training_step(sess, feed_dict={}):
        _, _loss = sess.run([train_step, loss], feed_dict=feed_dict)
        return _loss
    
    return layer_features, ae_output, loss, train_step, run_training_step
    
def train_seq2_seq_ae(data_dir, experiment_name, checkpoint_dir, log_dir, batch_size, \
                learning_rate, num_epochs, gpu_usage, tag, best_model_tag,
                max_seq_length = 430, num_channels=1025):
                
    g = tf.Graph()
    with g.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=config) 
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        saver = tf.train.Saver(var_list= None, max_to_keep=20)
                
        input_batch_placeholder = tf.placeholder(tf.float32, 
                    shape=(batch_size, max_seq_length, num_channels), name="input_batch_placeholder")
        seq_lengths_placeholder = tf.placeholder(tf.float32, 
                    shape=(batch_size,), name="seq_lengths_placeholder")
                    
        layer_features, ae_output, loss, train_step, run_training_step = \
            setup_seq2seq_ae(input_batch_placeholder, seq_lengths_placeholder, \
                            learning_rate=learning_rate, is_training=True, \
                            global_step=global_step)
                            
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            print("Start epoch {}, {}".format(epoch, experiment_name))

            batch_iterator = read_data_dir(data_dir, batch_size, shuffle=True, 
            allow_smaller_last_batch=False, fix_length=max_seq_length, 
            file_formats=["wav", "mp3"], error_on_different_fs=True)
            
            for step_batch, step_sequence_lengths, step_fs in batch_iterator:
                feed_dict = {input_batch_placeholder : step_batch,
                             seq_lengths_placeholder : step_sequence_lengths}
                step_loss = run_training_step(sess, feed_dict=feed_dict) 
                print("Epoch {} of {}.  Step loss {} .".format(epoch, num_epochs, step_loss))
                
            print("Saving model")
            save(sess, saver, checkpoint_dir, experiment_name, sess.run(global_step), tag=tag)    

    return layer_features, ae_output
