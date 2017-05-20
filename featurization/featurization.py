import librosa
from glob import glob

# https://github.com/DmitryUlyanov/neural-style-audio-tf
# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs
    
def invert_spectrogram(spectrogram):
    raise NotImplementedError()