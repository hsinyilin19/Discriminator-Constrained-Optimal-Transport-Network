import librosa
import os
import numpy as np
import scipy
import torch
from tqdm import tqdm
# import pdb, mkl
from util import check_folder
epsilon = np.finfo(float).eps
def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None, SHIFT=None, _max=None, _min=None):
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

    ### Normalize waveform
    # y = y / np.max(abs(y)) / 2.

    D = librosa.stft(y,center=False, n_fft=512, hop_length=160,win_length=512,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1

    return Sxx, phase, len(y)


def check_dir(path):
    if not os.path.isdir('/'.join(list(path.split('/')[:-1]))):
        os.makedirs('/'.join(list(path.split('/')[:-1])))

if __name__ == '__main__':
    for t in ['train', 'test']:
        print(f'converting {t}')
        train_path = f'/work/u8057472/VCTK_DEMAND/{t}/'
        train_convert_save_path = f'/work/u8057472/VCTK_DEMAND_pt/{t}/'

        # mkl.set_num_threads(1)
        n_frame = 64
        wav_files = get_filepaths(train_path)
        for wav_file in tqdm(wav_files):
            wav, sr = librosa.load(wav_file, sr=16000)
            out_path = wav_file.replace(train_path, train_convert_save_path).split('.w')[0]
            data, _, _ = make_spectrum(y=wav)
            for i in np.arange(data.shape[1]//n_frame):
                out_name = out_path + '_' + str(i) + '.pt'
                check_folder(out_name)
                torch.save(torch.from_numpy(data.transpose()[i * n_frame:(i+1) * n_frame]), out_name)
