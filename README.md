# DOTN: Discriminator-Constrained Optimal Transport Network

This repository hosts the Pytorch codes for paper Unsupervised Noise Adaptive Speech Enhancement by Discriminator-Constrained Optimal Transport by Hsin-Yi Lin, Huan-Hsin Tseng, Xugang Lu and Yu Tsao.

## Model


DOTN performs unsupervised domain adaptation for speech enhancement (SE), using optimal transport (OT) for domain alignment and Wasserstein Generative Adversarial Network (WGAN) to goven the output speech quality. 


## Datasets

### 1. Voice Bank corpus (VCTK)

[VCTK](https://datashare.is.ed.ac.uk/handle/10283/3443) includes speech utterances by 110 English speakers with various accents.

### 2. DEMAND

[DEMAND](https://zenodo.org/record/1227121#.YXgqnr_MKYY) contains 16-channel environmental noise recordings.


### 3. TIMIT

[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) contains recordings of 630 speakers of eight major dialects of American English.

## Prerequisites
- [Python 3.8](https://www.python.org/)
- [PyTorch 1.8.1+cu111](https://pytorch.org/)



## Hardware
- 1 GPU of 32 GB CUDA memory and 4 CPUs with 90 GB memory.

