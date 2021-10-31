import librosa, os
import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
import scipy

epsilon = np.finfo(float).eps


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)


def cal_score(clean, enhanced):
    clean = clean / abs(clean).max()
    enhanced = enhanced / abs(enhanced).max()
    s_stoi = stoi(clean, enhanced, 16000)
    s_pesq = pesq(clean, enhanced, 16000)

    return round(s_pesq, 5), round(s_stoi, 5)


def get_filepaths(directory, ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)


def make_spectrum(filename=None, y=None, feature_type='logmag'):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y / 32767.)
        elif y.dtype != 'float32':
            y = np.float32(y)

    D = librosa.stft(y, center=False, n_fft=512, hop_length=160, win_length=512, window=scipy.signal.hamming)
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # feature types
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D ** 2)
    else:
        Sxx = D

    return Sxx, phase, len(y)


def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10 ** Sxx_r)

    R = np.multiply(Sxx_r, phase)
    result = librosa.istft(R, center=False, hop_length=160, win_length=512, window=scipy.signal.hamming,
                           length=length_wav)
    return result
