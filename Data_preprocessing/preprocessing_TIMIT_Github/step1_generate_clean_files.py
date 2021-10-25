import os
import numpy as np
from sphfile import SPHFile   # pip install sphfile
from shutil import copyfile

TIMIT_path = "/Desktop/TIMIT"                          # TIMIT dataset location
target_root = "/Desktop/TIMIT_DA_target_helicopter"    # target folder for generated speech

def write_wav(list_file, write_path, outfile_name):
    infile = open(list_file, 'r')
    tgt_paths = []
    for line in infile:
        src_path = line.strip()
        sph = SPHFile(src_path)

        file = os.path.splitext(src_path.split('/')[-1])[0] + os.path.splitext(src_path.split('/')[-1])[1].lower()  # .WAV -> .wav
        tgt_path = os.path.join(write_path, src_path.split('/')[-2], file)

        if not os.path.exists(os.path.dirname(tgt_path)):
            os.makedirs(os.path.dirname(tgt_path))
        sph.write_wav(tgt_path)
        print('writing:', tgt_path)
        tgt_paths.append(tgt_path)

    # Write out the target list
    outfile = open(outfile_name, 'w')
    for file in tgt_paths:
        print(file, file=outfile)
    outfile.close()

    return tgt_paths


# the following test_spklist is suggested by original TIMIT core test set
test_spklist = [
   'MDAB0', 'MWBT0', 'FELC0',
   'MTAS1', 'MWEW0', 'FPAS0',
   'MJMP0', 'MLNT0', 'FPKT0',
   'MLLL0', 'MTLS0', 'FJLM0',
   'MBPM0', 'MKLT0', 'FNLP0',
   'MCMJ0', 'MJDH0', 'FMGD0',
   'MGRT0', 'MNJM0', 'FDHC0',
   'MJLN0', 'MPAM0', 'FMLD0'
]

# For Train, validation, and domain adaptation
train_list = []
valid_list = []
dat_list = []

test_speaker_folders = []
TRAIN_path = os.path.join(TIMIT_path, "TRAIN")
TEST_path = os.path.join(TIMIT_path, "TEST")


# select speakers among 8 Dialect Regions (called DR in TIMIT official)
for i in range(1, 9):
    # Collect speakers in "test_spklist"
    test_speaker_folders += [entry.path for entry in os.scandir(os.path.join(TEST_path, f'DR{i}')) if entry.name in test_spklist]

    # search for (total 24) male speakers in TRAIN folder
    r = os.path.join(TRAIN_path, f'DR{i}')
    speakers = [entry.name for entry in os.scandir(r) if entry.name.startswith('M')]
    speakers = sorted(speakers)

    # randomly select 10 male skeakers from "speakers"
    np.random.shuffle(speakers)
    for s in speakers[0:6]:
        train_list += [entry.path for entry in os.scandir(os.path.join(r, s)) if (entry.name.endswith('.WAV') and not entry.name.startswith('SA'))]   # SA sentances are excluded

    # search for (total 14) female speakers in TRAIN folder
    speakers = [entry.name for entry in os.scandir(r) if entry.name.startswith('F')]
    speakers = sorted(speakers)

    # randomly select 5 female skeakers from "speakers"
    np.random.shuffle(speakers)
    for s in speakers[:3]:
        train_list += [entry.path for entry in os.scandir(os.path.join(r, s)) if (entry.name.endswith('.WAV') and not entry.name.startswith('SA'))]  # SA sentances are excluded

test_list = [entry.path for folder in test_speaker_folders for entry in os.scandir(folder) if
             (entry.name.endswith('.WAV') and not entry.name.startswith('SA'))]

train_list = sorted(train_list)


# Write as Core Train List
outfile = open('CoreTrainList.txt', 'w')
for wav in train_list:
   print(wav, file=outfile)
outfile.close()


# Write as Core Test List
outfile = open('CoreTestList.txt', 'w')
for file in test_list:
   print(file, file=outfile)
outfile.close()


# convert Core Train, Test List to wav files
write_train_paths = write_wav(list_file='CoreTrainList.txt', write_path=os.path.join(target_root, 'train/clean/'), outfile_name='target_CoreTrainList.txt')
write_test_paths = write_wav(list_file='CoreTestList.txt', write_path=os.path.join(target_root, 'test/clean/'), outfile_name='target_CoreTestList.txt')
print('writing Train, Test lists done!')
