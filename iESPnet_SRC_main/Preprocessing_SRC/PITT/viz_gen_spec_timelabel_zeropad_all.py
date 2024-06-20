#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The spectrogram generated for each subject will be plto and saved on a pdf file
@author: vpeterson
April 2021
"""

#%%
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join('..', '/utilities')))
import IO # customized functions for navigating throught the folders and files
import Epochs
import torch
import torchaudio.transforms as T
import numpy as np
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal
import matplotlib.pyplot as plt
import librosa
from scipy import fft as sp_fft
from matplotlib.backends.backend_pdf import PdfPages
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
DATA_DIR = "/mnt/Nexus2/RNS_DataBank/PITT/" #change this path
ECOG_SAMPLE_RATE = 250

# For STFT
TT = 1000 # window length
overlap = 500 #
SPEC_WIN_LEN = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
SPEC_HOP_LEN = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000)   # Length of hop between windows.
# SPEC_WIN_LEN = 128 # win size
# SPEC_HOP_LEN = 125   # Le
SPEC_NFFT = 500  # to see changes of 0.5  Size of FFT, creates ``n_fft // 2 + 1`` bins
#%%
RNSIDS = IO.get_subfolders(DATA_DIR)
event_id = 1
tmin = 0.0
tmax = 90.0
tv = np.arange(0, int(tmax), 1/float(ECOG_SAMPLE_RATE))
top_db = 40.0
#%%
out_address = '/mnt/Nexus2/ESPnet/code/preprocessing/PITT/Preprocessing_plots/timelabel_zeropadall/' #change this path

if not os.path.exists(out_address):
    os.makedirs(out_address)
# read metada_data_files
# for the examplary purposes, only one subject here
for s in range(8,len(RNSIDS)):
    pp = PdfPages(out_address + RNSIDS[s] + '.pdf', keep_empty=False)

    # if RNSIDS[s] == 'PIT-RNS1529':  # this subject's data is useless
    #     continue
    annot_files = IO.get_annot_files(DATA_DIR, RNSIDS[s], Verbose=False)
    data_files = IO.get_data_files(DATA_DIR, RNSIDS[s], Verbose=False)


    if np.shape(data_files) != np.shape(annot_files):
        raise ValueError("Different number of files found")
        len(data_files)
    for nfile in range(len(data_files)):
        events = Epochs.get_events(annot_files[nfile])
        X, labels, times = Epochs.get_epochs_zeropad_all(data_files[nfile], events)
        # it could happen that all trials are short, and then no epochs have
        # been extracted.
        if len(X) == 0:
            continue
        # plot 3 examples
        # eof
        eof = np.where(labels == 0)[0]
        epoch = X[eof[0], 0, :]
        # normalize between -1 and 1
        # epoch = 2.*(epoch - np.min(epoch))/np.ptp(epoch)-1

        signal = torch.from_numpy(epoch)
        # normalize the waveform
        signal = (signal - signal.mean()) / signal.std()
        # signal_min, signal_max = signal.min(), signal.max()
        # signal_scaled = (signal - signal_min) / (signal_max - signal_min)
        # each channel is transformed independly
        spec, t, f = get_spectrogram(signal, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN)
        # spec to DB
        spec = librosa.power_to_db(spec, ref=np.median, top_db=top_db)
        idx_60 = np.where(f<=60)[0]
        spec = spec[:idx_60[-1],:]
        # label time
        label_time = np.zeros((spec.shape[1],))
        plt.figure()
        plt.subplot(211)
        plt.plot(tv,signal.numpy())
        plt.vlines(times[eof[0]], min(signal.numpy()), max(signal.numpy()), color='k')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(data_files[nfile][52:72] +' NO_SZ ' + ' epoch' + str(0))
        # plot spec
        plt.subplot(212)
        plt.imshow(spec, origin='lower', aspect='auto')
        plt.xticks(np.arange(0, len(t)-1,20), t[0:-1:20])
        f_60 = f[:idx_60[-1]]
        plt.yticks(np.arange(0, len(f_60)-1,20), f_60[0:-1:20])

        plt.savefig(pp, format='pdf')
        plt.close()
        # sz_on

        szon = np.where(labels == 1)[0]
        for l in range(len(szon)):
            epoch = X[szon[l], 0, :]
            # epoch = 2.*(epoch - np.min(epoch))/np.ptp(epoch)-1

            signal = torch.from_numpy(epoch)
            # normalize the waveform
            signal = (signal - signal.mean()) / signal.std()
            # signal_min, signal_max = signal.min(), signal.max()
            # signal_scaled = (signal - signal_min) / (signal_max - signal_min)

            # each channel is transformed independly
            spec, t, f = get_spectrogram(signal, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN)
            # spec to DB
            spec = librosa.power_to_db(spec, ref=np.median, top_db=top_db)
            idx_60 = np.where(f<= 60)[0]
            spec = spec[:idx_60[-1],:]
            spec = spec[:,:len(t)]
            # label time
            label_time = np.zeros((spec.shape[1],))
            label_time_smooth = np.zeros((spec.shape[1],))
            
            idx_t = np.where(t<=times[szon[l]])[0][-1]
            label_time[idx_t]=1
            # in very end
            if spec.shape[1]  - idx_t < 5:
                n = spec.shape[1]  - idx_t
                aux = np.arange(idx_t-n,idx_t+n,1)
            elif idx_t -5 <0 : # in the very begining
                n = 5 + (idx_t -5)
                aux = np.arange(idx_t-n,idx_t+n,1)

                    
            else:
                n = 5
                aux = np.arange(idx_t-n,idx_t+n,1)
            if aux.size!=0:                
                gaus =np.exp(-np.power(aux - idx_t, 2.) / (2 * np.power(1.5, 2.)))
                label_time_smooth[aux] = gaus
            else:
                label_time_smooth[idx_t] = 1
            plt.figure()
            plt.subplot(311)
            plt.plot(tv,signal.numpy())
            plt.vlines(times[szon[l]], min(signal.numpy()), max(signal.numpy()), color='k')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.autoscale(enable=True, axis='both', tight=True)

            plt.title(data_files[nfile][52:72] + ' SZ_ON ' +' epoch ' + str(l))
            # plot spec
            plt.subplot(312)
            plt.imshow(spec, origin='lower', aspect='auto')
            f_60 = f[:idx_60[-1]]
            plt.yticks(np.arange(0, len(f_60)-1,80), f_60[0:-1:80]);

            plt.xticks(np.arange(0, len(t)-1,40), t[0:-1:40]);
            plt.subplot(313)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.plot(label_time_smooth)
            plt.xticks(np.arange(0, len(t)-1,40), t[0:-1:40]);
            plt.savefig(pp, format='pdf')
            plt.close()


        # sz

        # sz = np.where(labels == 2)[0]
        # for l in range(len(sz)):
        #     epoch = X[sz[l], 0, :]
        #     # epoch = 2.*(epoch - np.min(epoch))/np.ptp(epoch)-1

        #     signal = torch.from_numpy(epoch)
        #     # normalize the waveform
        #     signal = (signal - signal.mean()) / signal.std()
        #     # each channel is transformed independly
        #     spec, t, f = get_spectrogram(signal, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN)
        #     # spec to DB
        #     spec = librosa.amplitude_to_db(spec, ref=1.0, amin=1e-8, top_db=top_db)
        #     idx_60 = np.where(f<= 60)
        #     spec = spec[:idx_60[-1],:]
        #     plt.figure()
        #     plt.subplot(211)
        #     plt.plot(tv,epoch)
        #     plt.vlines(times[sz[l]], min(epoch), max(epoch), color='k')
        #     plt.xlabel('Time')
        #     plt.ylabel('Amplitude')
        #     plt.title(data_files[nfile][52:72]+' SZ' + ' epoch ' + str(l))
        #     # plot spec
        #     plt.subplot(212)
        #     plt.imshow(spec, origin='lower', aspect='auto')
        #     plt.xticks(np.arange(0, len(t)-1,20), t[0:-1:20])
        #     # plt.savefig(pp, format='pdf')
        #     plt.close()

    pp.close()
