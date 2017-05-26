import tensorflow as tf
from layers import *
from ..featurization.featurization import read_data_dir
from utils import *

def conv_ae(input_batch, is_training, seed=12321):
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

    net = tf.layers.conv1d(net, 100, 3, strides=1, padding="SAME", use_bias=True, name="shrinking_layer",
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
    net = tf.nn.relu(net)
    shrink_layer_output = bn(net, is_training, "bn_shrinking_layer")

    # TODO double check that these conv1d_transposes are actually outputting the correct size    

    net = conv1d_transpose(shrink_layer_output, get_tf_shape_as_list(layer2), 256, window_size=3, 
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
    return net, [layer1, layer2, layer3, shrink_layer_output]
    
def conv_ae_with_loss(input_batch, training, seed=12321):
    """
    input_batch: The (B, T, F) shaped tensor representing audio.  B - batch size, T - time dimension, F - features
    seq_lengths: (b,) shape tensor containing the sequence masks
    training: If true, use mean of batch for batch norm, otherwise using moving average
    """
    ae_output, layer_features = conv_ae(input_batch, training, seed=seed)
    loss = tf.nn.l2_loss(input_batch - ae_output) / tf.cast(get_tf_shape_as_list(input_batch)[0], tf.float32)
    return ae_output, layer_features, loss

    
def setup_conv_ae(inputs, learning_rate=1e-3, is_training=True, global_step=None):
    # You can feed a placeholder for learning_rate if you want to use decay
    B, T, C = get_tf_shape_as_list(inputs)
    
    ae_output, layer_features, loss = conv_ae_with_loss(input_batch, is_training)
    
    solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = solver.minimize(loss, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    all_ops = update_ops + [train_step]
    train_step = tf.group(*all_ops)

    def run_training_step(sess, feed_dict={}):
        _, _loss = sess.run([train_step, loss], feed_dict=feed_dict)
        return _loss
    
    return layer_features, ae_output, loss, train_step, run_training_step
    
def train_conv_ae(data_dir, experiment_name, checkpoint_dir, log_dir, batch_size, \
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
                    
        layer_features, ae_output, loss, train_step, run_training_step = \
            setup_conv_ae(input_batch_placeholder, \
                            learning_rate=learning_rate, is_training=True, \
                            global_step=global_step)
                            
        for epoch in range(num_epochs):
            batch_iterator = read_data_dir(dir_path, batch_size, shuffle=True, 
            allow_smaller_last_batch=False, fix_length=max_seq_length, 
            file_formats=["wav", "mp3"], error_on_different_fs=True)
            
            for step_batch, step_sequence_lengths, step_fs in batch_iterator:
                feed_dict = {input_batch_placeholder : step_batch}
                step_loss = run_training_step(sess, feed_dict=feed_dict) 
                print("Epoch {} of {}.  Step loss {} .".format(epoch, num_epochs, step_loss))
                
            save(sess, saver, checkpoint_dir, experiment_name, sess.run(global_step), tag=tag)    

    return layer_features, ae_output