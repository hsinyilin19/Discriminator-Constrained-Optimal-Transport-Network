import os
import numpy as np
import librosa
import scipy.io.wavfile as wav
from shutil import copyfile
from sklearn.model_selection import train_test_split
import pathlib
import pickle

SR = 16000
make_copy = True

# select source & target domain noise from DEMAND dataset
source_noise = ["TBUS", "TCAR", "TMETRO"]
target_noise = ["SCAFE", "STRAFFIC", "SPSQUARE"]


VCTK_path = '/mnt/Datasets/Corpus/noisy-vctk-16k'  # original VCTK path
noise_path = '/mnt/Datasets/Corpus/DEMAND/data/'   # provide noise path
save_folder = '/mnt/Datasets/Corpus/'              # folder saving generated audio

# original VCTK Train, Test folders
VCTK_train_path = os.path.join(VCTK_path, "clean_trainset_28spk_wav_16k/")
VCTK_test_path = os.path.join(VCTK_path, "clean_testset_wav_16k/")
source_domain = "_".join(source_noise)
target_domain = "_".join(target_noise)


# prepare folders for source & target domain data
paths = {}
for mode in ['train', 'test']:
    for s in ['clean', 'noisy']:
        path = os.path.join(save_folder, f"VCTK_DEMAND_{source_domain}_to_{target_domain}/{mode}/{s}/")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)  # prepare folder
        paths[f'{mode}_{s}'] = path


def add_noise(clean_files, folder, noise_info, SNRs, output_path):
    noise_data, noise_types = noise_info
    for file in clean_files:
        clean_path = os.path.join(folder, file)
        clean, _ = librosa.load(clean_path, sr=SR)

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
    wav.write(output_file, SR, np.int16(output * 32767))


def read_noise(noise_path, noise_types):
    ''' read noise wav files'''
    noise_data = []
    for noise in noise_types:
        if not noise.endswith('.wav'):
            noise = noise + '.wav'

        n, sr = librosa.load(os.path.join(noise_path, noise), sr=SR)
        assert sr == SR, "Sampling rate != %s" % SR
        noise_data.append(n)
    return noise_data, noise_types

with open('speaker_info.pickle', 'rb') as handle:
    speaker_info = pickle.load(handle)
with open('male_speakers.pickle', 'rb') as handle:
    M = pickle.load(handle)
with open('female_speakers.pickle', 'rb') as handle:
    F = pickle.load(handle)



'''---------------- Step 1: prepare clean data for source (train) & target (test) domain ----------------'''
train_clean_files, dat_clean_files = train_test_split([file.name for file in os.scandir(VCTK_train_path) if file.name.endswith('.wav')], test_size=0.3)

male_utterances = [[file.name for file in os.scandir(VCTK_train_path) if file.name.startswith(spk)] for spk in M]
male_utterances = [x for x in male_utterances if x]      # remove empty male speaker

female_utterances = [[file.name for file in os.scandir(VCTK_train_path) if file.name.startswith(spk)] for spk in F]
female_utterances = [x for x in female_utterances if x]  # remove empty female speaker

# data split
train_list_M = male_utterances
train_list_F = female_utterances


# mixing male & female speakers
train_clean_files = [file for spk in train_list_M for file in spk] + [file for spk in train_list_F for file in spk]
test_clean_files = [file.name for file in os.scandir(VCTK_test_path) if file.name.endswith('.wav')]

if make_copy:
    for file in train_clean_files:
        copyfile(os.path.join(VCTK_train_path, file), os.path.join(paths['train_clean'], file))
    print('train clean prepared')

    for file in test_clean_files:
        copyfile(os.path.join(VCTK_test_path, file), os.path.join(paths['test_clean'], file))
    print('test clean prepared')



'''---------------- Step 2: add noise to clean ----------------'''
# add noise to Training
add_noise(train_clean_files,
          folder=paths['train_clean'],
          noise_info=read_noise(noise_path, source_noise),
          SNRs=[-9, -6, -3, 0, 3, 6, 9],
          output_path=paths['train_noisy'])

# add noise to Test
add_noise(test_clean_files,
          folder=paths['test_clean'],
          noise_info=read_noise(noise_path, target_noise),
          SNRs=[-9, -6, -3, 0, 3, 6, 9],
          output_path=paths['test_noisy'])

