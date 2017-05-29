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

    n_fft = 2048                            # STFT window size. See librosa.core.stft docs
    input_samples = 300                     # Free to change this. 
    input_channels = 1 + n_fft/2            # Computed. See librosa.core.stft docs on stft return value

    fe_content_layer = 3                    # For feature extraction 
    fe_style_layers = (                     # For feature extraction
        (1, 0.5),
        (2, 0.5)
    )
    fe_model = "conv_autoencoder_2d"             # Feature extractor model name (see FEATURE_EXTRACTOR_MODELS)
    fe_checkpoint = "checkpoints/last_checkpoint/hyperspectral_resnet.model-519"# Dir for feature extractor checkpoint files

    gen_alpha = 1e-2                        # weight for content loss
    gen_beta = 1                            # (redundant) weight for style loss
    gen_optimizer = "adam"
    gen_learning_rate = 1000.0
    gen_iterations = 1000
    gen_initializer = "random"
    gen_model = "conv_ae_with_loss"                 # Model name (see FEATURE_EXTRACTOR_MODELS)
    gen_checkpoint = "checkpoints/last_checkpoint/hyperspectral_resnet.model-519"# Dir for feature extractor checkpoint files
    gen_content_layer = 3                    # For generating the output audio
    gen_style_layers = (                     # For generating the output audio
        (1, 0.5),
        (2, 0.5)
    )

    
def gram(mat, n):
    "Compute a gram matrix"
    return  tf.matmul(tf.transpose(mat), mat)  / n

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
        result = {"loss":[]}
        content_spectrogram, content_sr = self.read_audio(content_filename)
        style_spectrogram, style_sr = self.read_audio(style_filename)
        print("CONTENT SPECTRO SHAPE", content_spectrogram.shape)
        # Save content and style spectrograms
        source_content_features = self.extract_content_features(content_spectrogram)
        source_style_features = self.extract_style_features(style_spectrogram)
        # Reconstruct and save style and content
        source_style_grams = [gram(f, self.config.input_samples) for f in source_style_features]

        with tf.variable_scope("input"):
            if self.config.initializer == "random": # Maybe later we want to change this.
                init = np.random.randn(1, self.config.input_samples, self.config.input_channels).astype(np.float32)*1e-3
            x = tf.Variable(init, name="x")
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            session, outputs, layer_features, loss = self.get_model(
                self.config.gen_model,
                x,
                self.config.gen_checkpoint
            )
        gen_content_features = layer_features[self.config.gen_content_layer]
        content_loss = tf.nn.l2_loss(gen_content_features - source_style_features)

        gen_style_features = [tf.squeeze(layer_features[i], axis[0]) for i, w in self.config.get_style_layers]
        gen_style_grams = [gram(f, self.config.input_samples) for f in gen_style_features]
        gen_style_losses = [tf.nn.l2_loss(gen_sg - source_sg) for gen_sg, source_sg in zip(gen_style_grams, source_style_grams)]
        style_loss = sum([loss * w for loss, (i,w) in zip(gen_style_losses, self.config.gen_style_layers)])

        loss = self.config.gen_alpha * content_loss + self.config.gen_beta * style_loss
        optimizer = OPTIMIZERS[self.config.gen_optimizer](self.config.learning_rate)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "input")
        grads = optimizer.compute_gradients(loss, var_list=q_vars)
        train_op = optimizer.apply_gradients(grads)

        session.run(tf.global_variables_initializer())
        for i in range(self.config.gen_iterations):
            _, loss_i = session.run((train_op, loss))
            result["loss"].append(loss_i)
        result["spectrogram"] = x.eval(session=session).T

        # TODO: this is not really where we want to send this stuff
        # TODO save spectrogram images
        self.write_audio("{}.wav".format(self.config.experiment_name), result["spectrogram"], content_sr)
        with open("{}.json".format(self.config.experiment_name), 'w') as outf:
            json.dump(result, outf)

    def extract_style_features(self, spectrogram):
        "Feeds a spectrogram into the feature extractor model and returns features for all the style layers"
        return self.fe_session.run(
            [self.fe_layer_features[i] for i, w in self.config.fe_style_layers],
            feed_dict={self.input_batch_placeholder: (spectrogram.T,)}
        )
        
    def extract_content_features(self, spectrogram):
        "Feeds a spectrogram into the feature extractor model and returns the content layer's features"
        return self.fe_session.run(
            self.fe_layer_features[self.config.fe_content_layer],
            feed_dict={self.input_batch_placeholder: (spectrogram.T,)}
        )

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




