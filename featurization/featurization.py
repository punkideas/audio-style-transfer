import librosa
from glob import glob
import os
import numpy as np

np.random.seed(123)

# https://github.com/DmitryUlyanov/neural-style-audio-tf
# Reads wav file and produces spectrum
# Fourier phases are ignored
# TODO what are these numbers?
# fs is the sampling rate
N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs
    
def invert_spectrogram(spectrogram, fs):
    """
    spectrogram should be a [time/samples, channels] shaped tensor
    fs is the sampling rate
    """
    result = np.exp(spectrogram.T) - 1

    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(result.shape) - np.pi
    for i in range(500):
        S = result * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    return x
   
def save_spectrogram_as_audio(spectrogram, fs, path):
    """
    spectrogram shape is: (time, channels)
    """
    as_audio = invert_spectrogram(spectrogram, fs)
    if np.any(np.isnan(as_audio)) or np.any(np.isinf(as_audio)):
        raise Exception("Failed to convert to audio")
    librosa.output.write_wav(path, as_audio, fs)
 
def fit_time_dim_to_size(x, size):
    time_dim = x.shape[1]
    if time_dim > size:
        return x[:, :size, :]
    elif time_dim < size:
        extra_size = size - time_dim
        return np.pad(x, [(0,0), (0, extra_size), (0,0)], 'constant')
    else:
        return x
    
def read_data_dir(dir_path, batch_size, shuffle=True, allow_smaller_last_batch=False, 
                    fix_length=None, file_formats=["wav", "mp3"], error_on_different_fs=True):
    """
    dir_path: path to directory to find data to read
    batch_size: yields data in batches of this size
    shuffle: If true, shuffle the dataset
    allow_smaller_last_batch:  If true, the last batch is allowed to be a smaller size 
            than batch_size (in case the dataset is not divisible by batch_size)
    fix_length: If a integral value, will always return batches of this length, cutting off points if need be.
    file_format:  Will only look for files of these formats
    error_on_different_fs: If true, throw an error if the fs (sampling rate) value for different
                           audio sources in the dataset are different
    
    Returns:
    An iterator (not a list) of tuples containing batches of data.
            The first element of the tuple is the spectrogram tensor (batch_size, max_time_length, num_channels)
            The second element is the associated sequence lengths for each batch
            The third element is the associated sampling rates (fs)
    max_time_length is the max time length of the spectrograms in the batch
    """
    data_file_names = []
    for file_format in file_formats:
        data_file_names = data_file_names + glob(os.path.join(dir_path, "*." + file_format))  + glob(os.path.join(dir_path, "**/*." + file_format))
    if shuffle:
        np.random.shuffle(data_file_names)

    print("Found {} files in {}".format(len(data_file_names), dir_path))
        
    while len(data_file_names) > 0:
        batch = data_file_names[:batch_size]
        data_file_names = data_file_names[batch_size:]
        if (not allow_smaller_last_batch) and len(batch) < batch_size:
            return
        # Read file names and transpose
        batch = [read_audio_spectum(f_name) for f_name in batch]
        fs = [data_point[1] for data_point in batch]
        batch = [data_point[0].T for data_point in batch]
        if error_on_different_fs and (not np.all(np.array(fs) == fs[0])):
            raise Exception("Some samples in the batch have different sampling rates")
        # Get sequence lengths and max length        
        sequence_lengths = np.array([data_point.shape[0] for data_point in batch])
        max_time_dimension = sequence_lengths.max()
        # Append batch dimension and concatenate
        time_length = fix_length
        if fix_length is None:
            time_length = max_time_dimension
        batch = [np.expand_dims(data_point, axis=0) for data_point in batch]
        batch = [fit_time_dim_to_size(data_point, time_length) for data_point in batch]
        sequence_lengths = np.minimum(sequence_lengths, time_length)
        batch = np.concatenate(batch, axis=0)
        yield batch, sequence_lengths, fs
    
if __name__ == "__main__":
    # Unit test
    for batch, seq_lengths, fs in read_data_dir("inputs/original_inputs/", 2):
        print(batch.shape)
        # print(batch)
    
