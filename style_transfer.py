import tensorflow as tf
import librosa
from models.autoencoder import * 
from models.conv_autoencoder_2d import conv_ae_with_loss
from models.utils import load, print_number_of_parameters
import os
import numpy as np
#import matplotlib.pyplot as plt
import json

FEATURE_EXTRACTOR_MODELS = {
    # More could be added here, so long as they are functions conforming to the API
    # outputs, layers, loss = fn(input_placeholder, seq_lengths, training)
    "seq2seq_ae": seq2seq_ae_with_loss,
    "conv_autoencoder_2d": conv_ae_with_loss
}

OPTIMIZERS = {
    "adam": tf.train.AdamOptimizer
}

class Config():
    """
    Fully specifies a style transfer pipeline. All keys and values must be JSON-serializable
    for distributed testing.
    """
    experiment_name = "experiment"
    n_fft = 2048                            
    input_samples = 300                     
    input_channels = 1 + n_fft/2            
    content_layer = 3                    
    style_layers = (                    
        (1, 0.5),
        (2, 0.5)
    )
    alpha = 1e-2                        
    beta = 1                            
    optimizer = "adam"
    learning_rate = 1000.0
    iterations = 1000
    white_noise_magnitude = 1e-3
    fe_model = "conv_autoencoder_2d"             
    fe_checkpoint = "checkpoints/last_checkpoint/hyperspectral_resnet.model-519"
    gen_model = "conv_ae_with_loss"                 
    gen_checkpoint = "checkpoints/last_checkpoint/hyperspectral_resnet.model-519"
    
class StyleTransferError(Exception):
    pass

class StyleTransfer():

    def __init__(self, config):
        """
        Assuming the feature extractor has already been trained, loads a TF checkpoint
        to initialize the feature extractor
        """
        self.config = config
        self.init_feature_extractor()

    def transfer_style(self, content_filename, style_filename):
        """
        Transfers style from `style_filename` onto content from `content_filename`
        by training the input to a model so that its content and style are as close
        to the respective content and style of the sources.
        """
        content_spectrogram, content_sr = self.read_audio(content_filename)
        style_spectrogram, style_sr = self.read_audio(style_filename)
        source_content_features = self.extract_content_features(content_spectrogram)
        source_style_features = self.extract_style_features(style_spectrogram)
            
        with tf.variable_scope('', reuse=False):
            x = tf.Variable(self.white_noise(), name="x")
        with tf.variable_scope('', reuse=True):
            #session, outputs, layer_features, loss = self.get_model(self.config.gen_model,
                    #x, self.config.gen_checkpoint)
            # This is hacky; it forces us to reuse the same model
            model = FEATURE_EXTRACTOR_MODELS[self.config.fe_model]
            outputs, layer_features, loss = model(x, training=False)
        gen_content_features = layer_features[self.config.content_layer]
        gen_style_features = [layer_features[i] for i, w in self.config.style_layers]
        content_loss = self.content_loss(source_content_features, gen_content_features)
        style_loss = self.style_loss(source_style_features, gen_style_features)
        loss = self.config.alpha * content_loss + self.config.beta * style_loss
        optimizer = OPTIMIZERS[self.config.optimizer](self.config.learning_rate)
        train_op = optimizer.minimize(loss, var_list=[x])

        self.fe_session.run(tf.global_variables_initializer())
        result = {"loss":[]}
        for i in range(self.config.iterations):
            _, loss_i = self.fe_session.run((train_op, loss))
            result["loss"].append(loss_i)
            print("i: {} of {}; loss = {}".format(i, self.config.iterations, loss_i))
        result["spectrogram"] = x.eval(session=session).T

        # TODO: this is not really where we want to send this stuff
        # TODO save spectrogram images
        self.write_audio("{}.wav".format(self.config.experiment_name), result["spectrogram"], content_sr)
        with open("{}.json".format(self.config.experiment_name), 'w') as outf:
            json.dump(result, outf)

    def extract_style_features(self, spectrogram):
        "Feeds a spectrogram into the feature extractor model and returns features for all the style layers"
        return self.fe_session.run(
            [self.fe_layer_features[i] for i, w in self.config.style_layers],
            feed_dict={self.input_batch_placeholder: (spectrogram.T,)}
        )
        
    def extract_content_features(self, spectrogram):
        "Feeds a spectrogram into the feature extractor model and returns the content layer's features"
        return self.fe_session.run(
            self.fe_layer_features[self.config.content_layer],
            feed_dict={self.input_batch_placeholder: (spectrogram.T,)}
        )

    def content_loss(self, source, generated):
        "Computes the content loss as the l2 norm of the difference between generated and source"
        return tf.nn.l2_loss(generated - source)

    def style_loss(self, source, generated):
        """
        Given lists of tensors `source` and `generated`, each of size 
        (batch_size, time_dim, channels_dim, num_filters),
        computes style loss as the l2 norm of the filter correlations, with a weight on each
        level's contribution.
        
        In the style transfer paper, they do not optimize for a match on values of filters within
        the style layer(s); they match correlations between average filter values over the image.
        See (Gatys et al, 2015, p. 10-11)
        """
        batch_size, time_dim, channels_dim, num_filters = source[0].shape
        N = num_filters
        M = time_dim * channels_dim
        return sum([
            tf.nn.l2_loss(self.filter_corrs(g) - self.filter_corrs(s)) * w
            for g, s, (i, w) in zip(generated, source, self.config.style_layers)
        ]) / (2*N**2*M**2)

    def filter_corrs(self, F):
        """
        Given a tensor F of size (batch_size, time_dim, channels_dim, num_filters),
        returns a filter G of size (batch_size, num_filters, num_filters) where G_ij
        is the (unscaled) correlation between filter i and filter j over the image. 
        """
        batch_size, time_dim, channels_dim, num_filters = [int(d) for d in F.shape]
        F_unrolled = tf.reshape(F, (batch_size, time_dim * channels_dim, num_filters))
        return tf.matmul(F_unrolled, F_unrolled, transpose_a=True)

    # TODO Specify basis for reconstruction. Currently, we just take the autoencoder output. 
    # but we could also synthesize just style or just content by training the gen autoencoder
    # on a loss function that only includes style or content loss. This could be helpful for
    # our paper. 
    # This process is described on page 10 of the style transfer paper.
    def reconstruct_spectrogram(self, spectrogram):
        "Returns a reconstruction of the spectrogram after passing through the feature extractor autoencoder"
        reconstruction =  self.fe_session.run(
            self.fe_outputs, 
            feed_dict={self.input_batch_placeholder: [spectrogram.T]}
        )
        return np.transpose(reconstruction, (0, 2, 1))
        
    def init_feature_extractor(self):
        """
        Instantiates the specified model with a specified checkpoint file. This only needs to happen on __init__, 
        so TF objects are stored as properties for reuse.
        """
        self.input_batch_placeholder = tf.placeholder(
            tf.float32, name="input_batch",
            shape=(1, self.config.input_samples, self.config.input_channels), 
        ) 
        self.fe_session, self.fe_outputs, self.fe_layer_features, self.fe_loss = self.get_model(
            self.config.fe_model,
            self.input_batch_placeholder,
            self.config.fe_checkpoint
        )

    def get_model(self, model_name, model_input, checkpoint):
        """
        Returns a model loaded with checkpoint data. Checkpoint should be something like
        "checkpoints/last_checkpoint/hyperspectral_resnet.model-519"
        """
        session = tf.Session()
        model = FEATURE_EXTRACTOR_MODELS[self.config.fe_model]
        outputs, layer_features, loss = model(model_input, training=False)
        saver = tf.train.Saver()
        saver.restore(session, checkpoint)
        return session, outputs, layer_features, loss

    def read_audio(self, filename):
        """
        Reads an audio file and converts it into a logarithmic spectrogram. Return value is 
        a numpy array of shape (self.config.input_channels, self.config.input_samples)
        """
        x, sample_rate = librosa.load(filename)
        S = librosa.stft(x, self.config.n_fft)
        p = np.angle(S)
        S = np.log1p(np.abs(S[:,:self.config.input_samples]))  
        required_padding = max(0, self.config.input_samples - S.shape[1])
        S = np.pad(S, ((0,0), (0, required_padding)), "constant")
        return np.array(S), sample_rate

    def write_audio(self, filename, spectrogram, sample_rate):
        """
        Reconstructs phase from spectrogram and writes the resulting audio to a file.
        Ensure that spectrograms passed in here have the correct orientation.
        """
        a = np.zeros_like(spectrogram)
        a[:self.config.input_channels,:] = np.exp(spectrogram[1]) - 1

        # This code is supposed to do phase reconstruction
        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
        for i in range(500):
            S = a * np.exp(1j*p)
            x = librosa.istft(S)
            p = np.angle(librosa.stft(x, N_FFT))

        librosa.output.write_wav(filename, x, sample_rate)

    def white_noise(self):
        """
        Constructs an array of white noise with which to initialize the input variable
        This noise will be transformed into the output audio.
        """
        shape = (1, self.config.input_samples, self.config.input_channels)
        return np.random.randn(*shape).astype(np.float32) * self.config.white_noise_magnitude




