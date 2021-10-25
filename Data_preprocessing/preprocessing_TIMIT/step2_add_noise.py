import os
import numpy as np
import librosa
import scipy.io.wavfile as wav

SR = 16000   # sampling rate
target_root = "/Desktop/TIMIT_DA_target_helicopter"

def add_noise(clean_files, target_folder, noise_info, SNRs):
    noise_data, noise_types = noise_info
    out_files = []
    for line in clean_files:
        clean, _ = librosa.load(line.strip(), sr=SR)
        line = line.strip().split('/')
        speaker = line[-2]
        file = os.path.splitext(line[-1])[0] + os.path.splitext(line[-1])[1].lower()  # .WAV -> .wav

        for i, noise in enumerate(noise_data):
            for snr in SNRs:
                out_file = generate_noisy(noise, clean, snr, target_folder, noise_types[i], speaker, file)
                out_files.append(out_file)
    return out_files


def generate_noisy(noise, clean, SNR, root, noise_name, speaker, file):
    out_path = os.path.join(root, noise_name, str(SNR) + 'dB', speaker, file)
    print('generating:', out_path)

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

    if max(abs(output)) > 1.:
        output /= max(abs(output))
        print(out_path)
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    wav.write(out_path, SR, np.int16(output * 32767))

    return out_path


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


'''---------------- Add noise to clean speech----------------'''

# add "stationary" noise to Training
train_noisy_files = add_noise(open('target_CoreTrainList.txt', 'r'),
          target_folder=os.path.join(target_root, 'train/noisy/'),
          noise_info=read_noise('./noise_types/stationary', ['car', 'engine',  'pink', 'wind1', 'cabin']),
          SNRs=[-12, -9, -6, -3, 0, 3, 6, 9, 12])


# add noise to Test
test_noisy_files = add_noise(open('target_CoreTestList.txt', 'r'),
          target_folder=os.path.join(target_root, 'test/noisy/'),
          noise_info=read_noise('./noise_types/nonstationary', ['helicopter']),
          SNRs=[-12, -9, -6, -3, 0, 3, 6, 9])


# write list
outfile = open('train_noisy_list.txt', 'w')
for file in train_noisy_files:
   print(file, file=outfile)
outfile.close()


# write list
outfile = open('test_noisy_list.txt', 'w')
for file in test_noisy_files:
   print(file, file=outfile)
outfile.close()
