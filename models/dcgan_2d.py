import tensorflow as tf
from layers import *
from featurization.featurization import read_data_dir, save_spectrogram_as_audio
from utils import *

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot

def DCGANGenerator(n_samples, T, C, noise=None, dim=64, bn=True, nonlinearity=tf.nn.relu, name="generator"):
    with tf.variable_scope(name):
        # https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
        
        assert T % 16 == 0
        assert C % 16 == 0
        start_T = T // 16
        start_C = C // 16

        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, start_T*start_C*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, start_T, start_C])
        if bn:
            output = Batchnorm('Generator.BN1', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN2', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN3', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN4', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 1, 5, output)
        # output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.squeeze(tf.transpose(output, [0,2,3,1]), axis=[3]), tf.ones((n_samples,)) * T

def discriminator(d_in, seq_lengths, is_training, name="discriminator", seed=2376):
    with tf.variable_scope(name):
        p = "VALID"

        layer1 = tf.layers.conv2d(d_in, 256, 11, strides=1, padding=p, use_bias=True, name="layer1",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer1 = leaky_relu(layer1)
        net = bn(layer1, is_training, "bn1")
        
        layer2 = tf.layers.conv2d(net, 256, 5, strides=2, padding=p, use_bias=True, name="layer2",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer2 = leaky_relu(layer2)
        net = bn(layer2, is_training, "bn2")
        
        layer3 = tf.layers.conv2d(net, 256, 3, strides=2, padding=p, use_bias=True, name="layer3",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer3 = leaky_relu(layer3)
        net = bn(layer3, is_training, "bn3")
        
        layer4 = tf.layers.conv2d(net, 256, 3, strides=2, padding=p, use_bias=True, name="layer4",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer4 = leaky_relu(layer4)
        net = bn(layer4, is_training, "bn4")  
        
        layer5 = tf.layers.conv2d(net, 256, 3, strides=2, padding=p, use_bias=True, name="layer5",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        layer5 = leaky_relu(layer5)
        net = bn(layer5, is_training, "bn4")

        net = tf.reshape(net, [-1] + [np.prod(get_tf_shape_as_list(net)[1:])])
        net = tf.layers.dense(net, 1)
        
        return net, [layer1, layer2, layer3, layer4, layer5]

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
    B, T, C = get_tf_shape_as_list(inputs
    G_sample, G_seq_lengths = DCGANGenerator(B, T, C)
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
                min_seq_length = 96, max_seq_length = 96, num_channels=1025):
                
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
                    
        style_transfer_feature_maps, G_sample, D_loss, G_loss, \
                D_train_step, G_train_step, run_training_step = \
            setup_gan(input_batch_placeholder, seq_lengths_placeholder)

        sess.run(tf.global_variables_initializer())
                            
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
    
    
    
    
