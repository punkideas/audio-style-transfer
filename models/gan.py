import tensorflow as tf
from layers import *
from featurization.featurization import read_data_dir, save_spectrogram_as_audio
from utils import *

def generator(sample_noise, output_dim, is_training, name="generator", seed=3571):
    """
    sample_noise should be a (B, T, D) tensor
    Will return a randomly sampled audio utterance of size (B, T, output_dim)
    """
    with tf.variable_scope(name):
        B, T, D = get_tf_shape_as_list(sample_noise)
        seq_lengths = tf.ones((B,)) * T
        original_seq_lengths = seq_lengths
        p = "VALID"

        layer1 = tf.layers.conv1d(sample_noise, 256, 11, strides=1, padding=p, use_bias=True, name="layer1",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer1 = tf.nn.relu(layer1)
        seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 11 + 1) / 1.)
        net = bn(layer1, is_training, "bn1")

        layer2 = tf.layers.conv1d(net, 256, 5, strides=2, padding=p, use_bias=True, name="layer2",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer2 = tf.nn.relu(layer2)
        seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 5 + 1) / 2.)
        net = bn(layer2, is_training, "bn2")

        with tf.variable_scope("lstm_1"):
            net, _ = lstm_layer(net, 512, seq_lengths)
        with tf.variable_scope("lstm_2"):
            net, _ = lstm_layer(net, 512, seq_lengths)

        net = conv1d_transpose(net, get_tf_shape_as_list(layer1), 256, window_size=5, stride=2, 
                               padding=p, use_bias=True, name="layer_d1")
        net = tf.nn.relu(net)
        net = bn(net, is_training, "bn_d1")

        net = conv1d_transpose(net, [B, T, 256], 256, window_size=11, stride=1, 
                               padding=p, use_bias=True, name="layer_d2")
        net = tf.nn.relu(net)
        net = tf.layers.conv1d(net, output_dim, 1, strides=1, padding=p, use_bias=True, name="layer_d3",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        # TODO usually end in tanh, but this requires outputs to be normalized into the range -1 to 1
        return net, original_seq_lengths

def discriminator(d_in, seq_lengths, is_training, name="discriminator", seed=2376):
    with tf.variable_scope(name):
        original_seq_lengths = seq_lengths
        p = "VALID"

        layer1 = tf.layers.conv1d(d_in, 256, 11, strides=1, padding=p, use_bias=True, name="layer1",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer1 = leaky_relu(layer1)
        seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 11 + 1) / 1.)
        net = bn(layer1, is_training, "bn1")
        
        layer2 = tf.layers.conv1d(net, 256, 5, strides=2, padding=p, use_bias=True, name="layer2",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer2 = leaky_relu(layer2)
        seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 5 + 1) / 2.)
        net = bn(layer2, is_training, "bn2")
        
        layer3 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=True, name="layer3",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer3 = leaky_relu(layer3)
        seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 3 + 1) / 2.)
        net = bn(layer3, is_training, "bn3")
        
        layer4 = tf.layers.conv1d(net, 256, 3, strides=2, padding=p, use_bias=True, name="layer4",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer4 = leaky_relu(layer4)
        seq_lengths = tf.ceil((tf.cast(seq_lengths, tf.float32) - 3 + 1) / 2.)
        net = bn(layer4, is_training, "bn4")

        encoder_outputs, encoder_final_state_tuple = unrolled_lstm_layer(net, 512, seq_lengths)
        net = tf.layers.dense(encoder_final_state_tuple.h, 1)
        return net, [layer1, layer2, layer3, layer4, encoder_final_state_tuple.c]

def wgan_loss(logits_real, logits_fake, batch_size, x, G_sample, x_seq_lengths):
    """Compute the WGAN-GP loss.
    TODO how to deal with different sequence lengths in generator and real input?
    # https://arxiv.org/abs/1704.00028
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    - batch_size: The number of examples in this batch
    - x: the input (real) images for this batch
    - G_sample: the generated (fake) images for this batch
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # compute D_loss and G_loss
    D_loss = tf.reduce_mean((logits_fake) - (logits_real))
    G_loss = tf.reduce_mean(- (logits_fake))

    # lambda from the paper
    lam = 10
    
    # random sample of batch_size (tf.random_uniform)
    eps = tf.random_uniform((batch_size,1, 1), minval=0, maxval=1)
    x_hat = (1 - eps) * G_sample + eps * x

    # Gradients of Gradients is kind of tricky!
    with tf.variable_scope(tf.get_variable_scope(),reuse=True) as scope:
        x_hat_logits, _ = discriminator(x_hat, x_seq_lengths, is_training=True)
        grad_D_x_hat = tf.gradients(x_hat_logits, [x_hat])[0]

    grad_norm = tf.sqrt(tf.reduce_sum(grad_D_x_hat * grad_D_x_hat, axis=1))
    grad_pen = grad_norm - 1
    grad_pen = grad_pen * grad_pen

    D_loss += lam * tf.reduce_mean(grad_pen)

    return D_loss, G_loss
    
def setup_gan(inputs, seq_lengths):
    B, T, C = get_tf_shape_as_list(inputs)
    sample_noise_dim = 100
    sample_noise = tf.random_uniform((B,T,sample_noise_dim), minval=-1, maxval=1)
    G_sample, G_seq_lengths = generator(sample_noise, C, is_training=True)
    real_logits, style_transfer_feature_maps = discriminator(inputs, seq_lengths, is_training=True)
    with tf.variable_scope(tf.get_variable_scope(),reuse=True) as scope:
        fake_logits, _ = discriminator(G_sample, G_seq_lengths, is_training=True)
    D_loss, G_loss = wgan_loss(real_logits, fake_logits, B, inputs, G_sample, seq_lengths)
    
    learning_rate = 1e-3
    beta1 = 0.5
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)  
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    D_all_ops = D_update_ops + [D_train_step]
    D_train_step = tf.group(*D_all_ops)

    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    G_all_ops = G_update_ops + [G_train_step]
    G_train_step = tf.group(*G_all_ops)

    def run_training_step(sess, feed_dicts=[{}, {}, {}, {}, {}], n_critic=5):
        assert len(feed_dicts) == n_critic
        for i in range(n_critic):
            _, d_loss = sess.run([D_train_step, D_loss], feed_dict=feed_dicts[i])
        _, g_loss = sess.run([G_train_step, G_loss], feed_dict=feed_dict[-1])
        return d_loss, g_loss
    
    return style_transfer_feature_maps, G_sample, D_loss, G_loss, \
                D_train_step, G_train_step, run_training_step

def train_gan(data_dir, experiment_name, checkpoint_dir, log_dir, batch_size, \
                learning_rate, num_epochs, gpu_usage, tag, best_model_tag,
                min_seq_length = 80, max_seq_length = 80, num_channels=1025):
                
    g = tf.Graph()
    with g.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=config) 
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        input_batch_placeholder = tf.placeholder(tf.float32, 
                    shape=(batch_size, max_seq_length, num_channels), name="input_batch_placeholder")
        seq_lengths_placeholder = tf.placeholder(tf.float32, 
                    shape=(batch_size,), name="seq_lengths_placeholder")
                    
        style_transfer_feature_maps, G_sample, D_loss, G_loss, \
                D_train_step, G_train_step, run_training_step = \
            setup_gan(input_batch_placeholder, seq_lengths_placeholder)

        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(var_list= None, max_to_keep=20)
                    
        step = 0
        for epoch in range(num_epochs):
            print("Start epoch {}, {}".format(epoch, experiment_name))

            batch_iterator = read_data_dir(data_dir, batch_size, shuffle=True, 
            allow_smaller_last_batch=False, fix_length=max_seq_length, 
            file_formats=["wav", "mp3"], error_on_different_fs=True)
            
            for step_batch, step_sequence_lengths, step_fs in batch_iterator:
                if np.any(step_sequence_lengths < min_seq_length):
                    continue

                step += 1
                feed_dict = {input_batch_placeholder : step_batch,
                             seq_lengths_placeholder : step_sequence_lengths}
                _, step_d_loss = sess.run([D_train_step, D_loss], feed_dict=feed_dict)
                step_g_loss = "Skipped this step"
                if step % 5 == 0:
                    _, step_g_loss = sess.run([G_train_step, G_loss], feed_dict=feed_dict)
                    
                print("Epoch {} of {}.  Step d_loss {}, step g_loss {} .".format(epoch, \
                            num_epochs, step_d_loss, step_g_loss))
                
            print("Saving model")
            save(sess, saver, checkpoint_dir, experiment_name, step, tag=tag)    

        sample_save_path = os.path.join(checkpoint_dir,  "samples")
        if not os.path.exists(sample_save_path):
            os.makedirs(sample_save_path)

        batch_iterator = read_data_dir(data_dir, batch_size, shuffle=True,
            allow_smaller_last_batch=False, fix_length=max_seq_length,
            file_formats=["wav", "mp3"], error_on_different_fs=True)

        for step_batch, step_sequence_lengths, step_fs in batch_iterator:
            feed_dict = {input_batch_placeholder : step_batch,
                             seq_lengths_placeholder : step_sequence_lengths}
            step_G_sample = sess.run(G_sample, feed_dict=feed_dict)

            for i in range(step_G_sample .shape[0]):
                spectrogram = step_G_sample [i, :, :]
                fs = step_fs[i]
                save_spectrogram_as_audio(spectrogram, fs, os.path.join(sample_save_path, str(i) + "_sample.wav"))

            break

    return style_transfer_feature_maps, G_sample
    
    
    
    
