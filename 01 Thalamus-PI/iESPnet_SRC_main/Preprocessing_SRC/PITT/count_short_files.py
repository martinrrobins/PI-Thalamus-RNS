#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this function we are going to read EDF files, annotations and generate
2D STFT images
This is  Pytorch based script
@author: vpeterson
April 2021
"""

#%%
WINDOWS = False
import os
import pandas as pd
import sys
if WINDOWS:
    sys.path.insert(1, 'C:/Users/vp820/Documents/GitHub/SeizConvNet/utilities')
else:
    sys.path.insert(1, '/mnt/Nexus2/ESPnet/code/utilities')

import IO # customized functions for navigating throught the folders and files
import Epochsv2
import torch
import torchaudio.transforms as T
import numpy as np
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal
import matplotlib.pyplot as plt
import librosa
from scipy import fft as sp_fft
from itertools import permutations
#%%
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    # fig.colorbar(im, ax=axs)
    plt.show(block=False)

def get_spectrogram(signal, fs, n_fft = 256, win_len = None, hop_len = None,
                    power = 2.0):
    wind_dic={
              'periodic':True,
              'beta':10}
    spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_len,
                                hop_length=hop_len, pad=0,
                                window_fn =torch.kaiser_window,
                                normalized=False,
                                power=power, wkwargs=wind_dic)
    time = np.arange(win_len/2, signal.shape[-1] + win_len/2 + 1,
                     win_len - (win_len-hop_len))/float(fs)
    time -= (win_len/2) / float(fs)
    freqs = sp_fft.rfftfreq(n_fft, 1/fs)
    return spectrogram(signal), time, freqs

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

#%% Define Data Parameters
# All this patients are PITT

if WINDOWS:
        
    # DATA_DIR = "/Y:/RNS_DataBank/PITT/"
    OUTDATA_DIR = "C:/Users/vp820/Documents/SPMnet/Data/TimeLabelpad/PITT/"
    OUTMETADATA_DIR = "C:/Users/vp820/Documents/SPMnet/Data/TimeLabelpad/PITT//METADATA/"
else:
    
    DATA_DIR = "/mnt/Nexus2/RNS_DataBank/PITT/"
    OUTDATA_DIR = "/mnt/Nexus2/ESPnet/RNS_Databank_Spectrograms/TimeLabelSpecZeropadAll/PITT/"
    OUTMETADATA_DIR = "/mnt/Nexus2/ESPnet/RNS_Databank_Spectrograms/TimeLabelSpecZeropadAll/PITT/METADATA/"
    

if not os.path.exists(OUTDATA_DIR):
    os.makedirs(OUTDATA_DIR)

if not os.path.exists(OUTMETADATA_DIR):
    os.makedirs(OUTMETADATA_DIR)
    
    
ECOG_SAMPLE_RATE = 250
ECOG_CHANNELS = 4

TT = 1000 # window length
overlap = 500 #
SPEC_WIN_LEN = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
SPEC_HOP_LEN = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000)   # Length of hop between windows.
# SPEC_WIN_LEN = 128 # win size
# SPEC_HOP_LEN = 125   # Le
SPEC_NFFT = 500  # to see changes in 0.5 reso
#%%
RNSIDS = IO.get_subfolders(DATA_DIR)
# channel-wise spectrogram will be stacked so as to construct a unique image
# for the sake of data augmentation, as many spectrogram will be saved as
# possible permutation between channel positions.
top_db = 40.0
NFREQ = 120
NTIME = 181
#%%
# read metada_data_files
df = pd.DataFrame(columns=['rns_id', 'data', 'label', 'time'])
counter_s = np.zeros((len(RNSIDS, )))
for s in range(len(RNSIDS)):
    counter = []

    print('Running subject ' + RNSIDS[s] + ' [s]: ' + str(s))
    # if RNSIDS[s] == 'PIT-RNS1529':  # this subject's data is useless
    #     break
    #     continue
    data_files = IO.get_data_files(DATA_DIR, RNSIDS[s], Verbose=False)
    annot_files = IO.get_annot_files(DATA_DIR, RNSIDS[s], Verbose=False)
    
    if np.shape(data_files) != np.shape(annot_files):
        raise ValueError("Different number of files found")
        len(data_files)
    # if not os.path.exists(OUTDATA_DIR+RNSIDS[s]):
    #     os.makedirs(OUTDATA_DIR+RNSIDS[s])
    for nfile in range(len(data_files)):
        events = Epochsv2.get_events(annot_files[nfile])
        idx = Epochsv2.get_short_files(events)
        
        counter.append(len(idx))
        
    counter_s[s] = sum(counter)
        
       

        
