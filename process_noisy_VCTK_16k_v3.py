import os
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import scipy.io.wavfile as wav
import random
import multiprocessing
from joblib import Parallel, delayed
import glob
from shutil import copyfile
from sklearn.model_selection import train_test_split
import pathlib
import pickle

np.random.seed(1234)

_SR = 16000
num_cores = 48
make_copy = True

def add_noise(clean_files, folder, noise_info, SNRs, output_path):
    noise_data, noise_types = noise_info
    for file in clean_files:
        clean_path = os.path.join(folder, file)
        clean, _ = librosa.load(clean_path, sr=_SR)

        for i, noise in enumerate(noise_data):
            for snr in SNRs:
                generate_noisy(noise, clean, snr, output_path, noise_types[i], file)


def generate_noisy(noise, clean, SNR, output_path, noise_name, clean_name):
    output_file = os.path.join(os.path.join(output_path, noise_name, str(SNR) + 'dB'), clean_name)
    print('generating ', output_file)

    '''if noise (duration) shorter than speech, repeat noise to match length of speech'''
    if len(noise) < len(clean):
        duplication = (len(clean) // len(noise)) + 1
        noise = np.concatenate([noise] * duplication)

    from_point = np.random.randint(0, len(noise) - len(clean) + 1, 1)[0]
    noise = noise[from_point:from_point + len(clean)]
    assert len(noise) == len(clean), 'len(noise) == len(clean)'

    s_pwr = np.var(clean)
    noise = noise - np.mean(noise)
    n_var = s_pwr / (10**(SNR / 10.))
    noise = np.sqrt(n_var) * noise / np.std(noise)
    output = clean + noise

    # print(10*np.log10(sum(s**2)/sum(noise**2)))
    if max(abs(output)) > 1.:
        output /= max(abs(output))
        print(output_file)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    wav.write(output_file, _SR, np.int16(output * 32767))


def read_noise(noise_path, noise_types):
    ''' read noise wav files'''
    noise_data = []
    for noise in noise_types:
        if not noise.endswith('.wav'):
            noise = noise + '.wav'

        n, sr = librosa.load(os.path.join(noise_path, noise), sr=_SR)
        assert sr == _SR, "Sampling rate != %s" % _SR
        noise_data.append(n)
    return noise_data, noise_types

with open('speaker_info.pickle', 'rb') as handle:
    speaker_info = pickle.load(handle)
with open('male_speakers.pickle', 'rb') as handle:
    M = pickle.load(handle)
with open('female_speakers.pickle', 'rb') as handle:
    F = pickle.load(handle)

# From original Train folder split for "train" & "domain adaptation"
original_train_path = "/work/u8057472/noisy-vctk-16k/clean_trainset_28spk_wav_16k/"
original_test_path = "/work/u8057472/noisy-vctk-16k/clean_testset_wav_16k/"
train_clean_files, dat_clean_files = train_test_split([file.name for file in os.scandir(original_train_path) if file.name.endswith('.wav')], test_size=0.3)

male_utterances = [[file.name for file in os.scandir(original_train_path) if file.name.startswith(spk)] for spk in M]
male_utterances = [x for x in male_utterances if x]   # remove empty male speaker

female_utterances = [[file.name for file in os.scandir(original_train_path) if file.name.startswith(spk)] for spk in F]
female_utterances = [x for x in female_utterances if x]  # remove empty female speaker

# data split
train_list_M = male_utterances
train_list_F = female_utterances

# prepare train clean
train_clean_path = "/work/u8057472/VCTK_DEMAND/train/clean/"
pathlib.Path(train_clean_path).mkdir(parents=True, exist_ok=True)

# mixing male & female speakers
train_clean_files = [file for spk in train_list_M for file in spk] + [file for spk in train_list_F for file in spk]

# prepare test clean
test_clean_path = "/work/u8057472/VCTK_DEMAND/test/clean/"
pathlib.Path(test_clean_path).mkdir(parents=True, exist_ok=True)
test_clean_files = [file.name for file in os.scandir(original_test_path) if file.name.endswith('.wav')]

if make_copy:
    for file in train_clean_files:
        copyfile(os.path.join(original_train_path, file), os.path.join(train_clean_path, file))
    print('train clean prepared')

    for file in test_clean_files:
        copyfile(os.path.join(original_test_path, file), os.path.join(test_clean_path, file))
    print('test clean prepared')


'''---------------- Add noise to clean ----------------'''
train_noisy_path = "/work/u8057472/VCTK_DEMAND/train/noisy/"
test_noisy_path = "/work/u8057472/VCTK_DEMAND/test/noisy/"

pathlib.Path(train_noisy_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(test_noisy_path).mkdir(parents=True, exist_ok=True)

# add noise to Training
add_noise(train_clean_files,
          folder=train_clean_path,
          noise_info=read_noise('/work/u8057472/DEMAND/data/', ["TBUS", "TCAR", "TMETRO"]),
          SNRs=[-9, -6, -3, 0, 3, 6, 9],
          output_path=train_noisy_path)

# add noise to Test
add_noise(test_clean_files,
          folder=test_clean_path,
          noise_info=read_noise('/work/u8057472/DEMAND/data/', ["SCAFE", "SPSQUARE", "STRAFFIC"]),
          SNRs=[-9, -6, -3, 0, 3, 6, 9],
          output_path=test_noisy_path)

