import librosa
import os
import numpy as np
import scipy
import torch
from tqdm import tqdm
from pathlib import Path

epsilon = np.finfo(float).eps
SR = 16000   # sampling rate

def get_filepaths(directory, ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)


def make_spectrum(filename=None, y=None, feature_type='logmag', _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    D = librosa.stft(y, center=False, n_fft=512, hop_length=160, win_length=512, window=scipy.signal.hamming)
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # select feature types
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    return Sxx, phase, len(y)


if __name__ == '__main__':
    target_root = "/mnt/Datasets/Corpus/VCTK_DEMAND_TBUS_TCAR_TMETRO_to_SCAFE_STRAFFIC_SPSQUARE"

    for stage in ['train', 'test']:
        print(f'converting {stage} files')
        train_path = os.path.join(target_root, stage)
        train_convert_save_path = os.path.join(target_root + '_pt', stage)

        n_frame = 64
        wav_files = get_filepaths(train_path)
        for wav_file in tqdm(wav_files):
            wav, sr = librosa.load(wav_file, sr=SR)
            out_path = wav_file.replace(train_path, train_convert_save_path).split('.w')[0]
            data, _, _ = make_spectrum(y=wav)
            for i in np.arange(data.shape[1] // n_frame):
                Path(out_path).mkdir(parents=True, exist_ok=True)
                out_name = out_path + '_' + str(i) + '.pt'
                torch.save(torch.from_numpy(data.transpose()[i * n_frame:(i + 1) * n_frame]), out_name)