# DOTN: Discriminator-Constrained Optimal Transport Network

This repository hosts the Pytorch codes for paper Unsupervised Noise Adaptive Speech Enhancement by Discriminator-Constrained Optimal Transport by Hsin-Yi Lin, Huan-Hsin Tseng, Xugang Lu and Yu Tsao.

## Model


DOTN performs unsupervised domain adaptation for speech enhancement (SE), using optimal transport (OT) for domain alignment and Wasserstein Generative Adversarial Network (WGAN) to goven the output speech quality. 


## Datasets & Preprocessing
###  - [Voice Bank corpus](https://datashare.ed.ac.uk/handle/10283/2791) (VCTK)

In `Data_preprocessing/processing_VCTK_Demand`:
1. Download [clean_trainset_28spk_wav](https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y) and [clean_testset_wav](https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y), two subsets of VCTK, and put them together in a larger folder, say `VCTK_noisy`.
2. Download [DEMAND](https://zenodo.org/record/1227121#.YXgqnr_MKYY) noise (16-channel environmental noise recordings)
3. Add paths of VCTK and DEMAND (noise) in `step1_process_noisy_VCTK_16k.py`
4. Convert generated *.wav* files to *.pt* files using `step2_convert_to_pt.py`

### - [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) Acoustic-Phonetic Continuous Speech Corpus
In `Data_preprocessing/preprocessing_TIMIT`:
1. Put TIMIT path in `step1_generate_clean_files.py` to generate clean speech
2. Add path of `noise_types` folder in `step2_add_noise.py ` to mix clean speech with noise
3. Convert generated *.wav* files to *.pt* files using `step3_convert_to_pt.py`

## Run Discriminator-Constrained Optimal Transport Network
Provide generated data paths **data_path** & **pt_data_path** in `main.py`, and run ```python main.py```


## Prerequisites
- [Python 3.8](https://www.python.org/)
- [PyTorch 1.8](https://pytorch.org/)
- [POT 0.8.0](https://pythonot.github.io/)
- [librosa 0.8.1](https://librosa.org/doc/latest/index.html)
- [pypesq 1.2.4](https://pypi.org/project/pypesq/)
- [pystoi 0.3.3](https://pypi.org/project/pystoi/)
- [Tensorboard 2.7.0](https://pypi.org/project/tensorboard/)
- [scikit-learn 1.0.1](https://pypi.org/project/scikit-learn/)
- [tqdm 4.62.3](https://pypi.org/project/tqdm/)


## Hardware
- NVIDIA V100 (32 GB CUDA memory) and 4 CPUs.
