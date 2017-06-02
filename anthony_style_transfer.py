from style_transfer import Config, StyleTransfer
from featurization.featurization import save_spectrogram_as_audio
import numpy as np

content_file = "inputs/vctk_corpus/VCTK-Corpus/wav48/val/p300/p300_024.wav"
style_file = "inputs/vctk_corpus/VCTK-Corpus/wav48/train/p228/p228_044.wav"
config = Config()
config.experiment_name = "anthony-st"
config.fe_checkpoint  = "saved_checkpoints/overfit_on_p228_2d_conv_ae2d_conv_ae/overfit_on_p228_2d_conv_ae2d_conv_ae/hyperspectral_resnet.model-519"
config.iterations = 80
config.decay_iteration = 60
config.optimizer = "adam"
config.learning_rate = 1.0
config.decayed_learning_rate = 0.0333
config.start_with_content = False
config.input_samples = 188
config.content_layer = 0
config.style_layers = (
        (0, 1.0),
        #(0, 0.5),
        #(1, 0.5)
    )
config.reg = 0.0
config.alpha = 1e-3

st = StyleTransfer(config)
out, out_sr = st.transfer_style(content_file, style_file)
print("shape: ", out.shape)
print("Contains nan: ", np.any(np.isnan(out)))
print("min: ", out.min())
print("max: ", out.max())
print("mean: ", out.mean())
print("std: ", out.std())

try:
    save_spectrogram_as_audio(out.T, out_sr, "log/" + config.experiment_name + "_out.wav")
except Exception as e:
    print(e)
    print("Trying clipping")
    out = np.clip(out, -40, 40)











