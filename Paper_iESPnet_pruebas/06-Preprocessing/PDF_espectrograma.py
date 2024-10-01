import pandas as pd
import sys
import os

sys.path.insert(1, '/home/martin/Documentos/PI-Thalamus/01 Thalamus-PI/iESPnet_SRC_main/utilities')
import IO # customized functions for navigating throught the folders and files
import Epochs
from Generator import smoothing_label

import torch
import torchaudio.transforms as T
import numpy as np
import mne
mne.set_log_level(verbose='warning') #to avoid info at terminal
import matplotlib.pyplot as plt
import librosa
import re
import matplotlib.font_manager as fm

from scipy import fft as sp_fft
from itertools import permutations
from utilit_espectrograms import get_data_files, get_annot_files, get_epochs_zeropad_all, get_events, get_patient_PE, get_spectrogram_1
from matplotlib.backends.backend_pdf import PdfPages

# Definición de variables para crear el espectrograma 

ECOG_SAMPLE_RATE = 250
ECOG_CHANNELS    = 4
TT               = 1000 # window length
SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
overlap          = 500 
SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
SPEC_NFFT        = 500  # to see changes in 0.5 reso
top_db           = 60.0

DATA_DIR = "/media/martin/Disco2/Rns_Data/RNS_ESPM_datatransfer/Data"
RNSIDS   = [
            #'PIT-RNS0427',
            'PIT-RNS1713',  
            'PIT-RNS3016',
            #'PIT-RNS7168',
            #'PIT-RNS8326'
           ]

out_address = '/media/martin/Disco2/Rns_Data/Representaciónes_PDF/'

if not os.path.exists(out_address):
    os.makedirs(out_address)

for s in range(len(RNSIDS)):
      
    pp = PdfPages(out_address + RNSIDS[s] + '.pdf', keep_empty=False)
    data_files  = get_data_files(DATA_DIR, RNSIDS[s], Verbose=False)
    annot_files = get_annot_files(DATA_DIR, RNSIDS[s], Verbose=False)

    pattern = re.compile(r'PE\d{8}-\d')
    extracted_parts = [pattern.search(path).group(0) for path in data_files]
    print('run patient: {}'.format(RNSIDS[s]))

    for nepoch in range(len(data_files)):

        events = get_events(annot_files[nepoch])
        X, labels, times = get_epochs_zeropad_all(data_files[nepoch], events)

        # Encontrar los índices donde labels == 1
        nfiles_labels_1 = np.where(labels == 1)[0]

        # Seleccionar aleatoriamente 5 archivos donde labels == 0
        nfiles_labels_0 = np.where(labels == 0)[0]
        if len(nfiles_labels_0) > 5:
            nfiles_labels_0 = np.random.choice(nfiles_labels_0, 5, replace=False)

        # Combinar los archivos donde labels == 1 con los archivos aleatorios donde labels == 0
        nfiles_to_include = np.concatenate((nfiles_labels_1, nfiles_labels_0))
        
        for nfile in nfiles_to_include:

            epoch = X[nfile, :, :]

            signal = torch.from_numpy(epoch)
            signal = (signal - signal.mean()) / signal.std()
            spec, t, f = get_spectrogram_1(signal, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN)

            spec = librosa.power_to_db(spec, top_db=top_db)
            idx_60 = np.where(f<= 60)[0][-1]
            spec = spec[:, :idx_60,:]
            
            # label time
            label_time = np.zeros((spec.shape[2],))
            if labels[nfile] !=0:
                idx_t = np.where(t<=times[nfile])[0][-1]
                label_time[idx_t]=1
            
            target_transform_1 = smoothing_label()
            label_1=target_transform_1(label_time)

            # Definición de variables para plot

            raw              = mne.io.read_raw_edf(data_files[nepoch])
            sf               = raw.info['sfreq'] # frecuencia de muestreo
            intervalo_tiempo = 1 / sf # segundos entre cada muestra
            t_1              = np.arange(signal.shape[1])
            tiempo           = t_1 * intervalo_tiempo
            ticks_x          = np.arange(0, tiempo.max()+1, 10)

            font_path= '../03-Letra-plot/Montserrat-Regular.ttf'
            montserrat = fm.FontProperties(fname=font_path)

            fig, axs = plt.subplots(3, 1, figsize=(6, 6), facecolor='#F2F2F2')

            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0.4)

            # iEEG
            axs[0].plot(tiempo, signal[0, :], color='#BEE6DC', linestyle='-', linewidth=0.5)
            axs[0].set_xticks(ticks_x)
            axs[0].set_title('{} - {}'.format(RNSIDS[s],extracted_parts[nepoch]), fontproperties=montserrat, fontsize=18)
            axs[0].set_ylabel('Amplitud (microV)', fontsize=12, fontproperties=montserrat,labelpad=25)
            axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#CFCFCF')
            axs[0].set_xlim(0, 90)
            axs[0].axvline(x=times[nfile], color='#8B97F4', linestyle='--', linewidth=1.5)

            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[0].spines['left'].set_visible(False)
            axs[0].spines['bottom'].set_visible(False)

            # Espectrograma
            axs[1].imshow(spec[0, :, :], origin='lower', aspect='auto')
            axs[1].set_ylabel('Frecuencia (Hz)', fontsize=12, fontproperties=montserrat,labelpad=25)
            axs[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#CFCFCF')

            # Etiquetas del eje X para que correspondan a 0-90 segundos
            num_xticks = 10
            x_tick_positions = np.linspace(0, spec.shape[2], num_xticks)
            x_tick_labels = [str(int(x * 90 / (spec.shape[2] - 1))) for x in x_tick_positions]
            axs[1].set_xticks(x_tick_positions)
            axs[1].set_xticklabels(x_tick_labels)

            # Etiquetas del eje Y para que correspondan a las frecuencias
            num_yticks = 6
            y_tick_positions = np.linspace(0, spec.shape[1], num_yticks)
            y_tick_labels = [str(int(y * 60 / (spec.shape[1] - 1))) for y in y_tick_positions]
            axs[1].set_yticks(y_tick_positions)
            axs[1].set_yticklabels(y_tick_labels)

            # Estilo de los ejes
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)
            axs[1].spines['left'].set_visible(False)
            axs[1].spines['bottom'].set_visible(False)

            # smoothing label
            axs[2].plot(label_1, color='#BEE6DC', linestyle='-', linewidth=2)
            axs[2].set_xlabel('Tiempo (segundos)', fontsize=12, fontproperties=montserrat)
            axs[2].set_ylabel('Probabilidad', fontsize=12, fontproperties=montserrat,labelpad=25)
            axs[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#CFCFCF')

            axs[2].set_xlim(0, 90)
            axs[2].set_xticks(x_tick_positions)
            axs[2].set_xticklabels(x_tick_labels)

            axs[2].spines['top'].set_visible(False)
            axs[2].spines['right'].set_visible(False)
            axs[2].spines['left'].set_visible(False)
            axs[2].spines['bottom'].set_visible(False)

            # Alinear los ylabel
            for ax in axs:
                ax.yaxis.set_label_coords(-0.12, 0.5)

            # Ajustar el diseño para evitar superposiciones
            fig.tight_layout()

            pp.savefig(fig)
            plt.close(fig)

    pp.close()