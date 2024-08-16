import pandas as pd
import sys
import os
import torch
import torchaudio.transforms as T
import numpy as np
import mne
import librosa
import utilit_espectrograms as ue

DATA_DIR        = '/media/martin/Disco2/Rns_Data/RNS_ESPM_datatransfer/Data'
OUTDATA_DIR     = '/media/martin/Disco2/Rns_Data/PITT_PI_EEG_PROCESS/'
OUTMETADATA_DIR = '/media/martin/Disco2/Rns_Data/PITT_PI_EEG_PROCESS/METADATA/'

# crear las carpetas en caso de que no existan

if not os.path.exists(OUTDATA_DIR):
    os.makedirs(OUTDATA_DIR)

if not os.path.exists(OUTMETADATA_DIR):
    os.makedirs(OUTMETADATA_DIR)


# lista con ids de pacientes

RNSIDS = ue.get_subfolders(DATA_DIR)

# df metadata

df = pd.DataFrame(columns=['rns_id', 'data', 'label', 'time'])

for s in range(len(RNSIDS)):
    
    print('Running subject ' + RNSIDS[s] + ' [s]: ' + str(s))
    data_files  = ue.get_data_files(DATA_DIR, RNSIDS[s], Verbose=False)
    annot_files = ue.get_annot_files(DATA_DIR, RNSIDS[s], Verbose=False)

    for nepoch in range(len(data_files)):

        events           = ue.get_events(annot_files[nepoch])
        X, labels, times = ue.get_epochs_zeropad_all(data_files[nepoch], events)

        [nt, nc, ns] = np.shape(X)        

        hosp_id, subject_id, PE_id = ue.get_patient_PE(data_files[nepoch], RNSIDS[s])

        for nfile in range(nt):
            file   = X[nfile, :, :]
            signal = torch.from_numpy(file)
            signal = (signal - signal.mean()) / signal.std()
            signal = signal.to(torch.float32)

            # Calcular los cuartiles y el rango intercuartil
            Q1 = signal.quantile(0.25)
            Q3 = signal.quantile(0.75)
            IQR = Q3 - Q1

            # Definir los límites para los valores atípicos
            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR

            outliers = (signal < lim_inf) | (signal > lim_sup)
            signal[outliers] = 0

            # label
            label_time = np.zeros(2)
            label_time[0] = labels[nfile]
            label_time[1] = times[nfile]
            label_time.astype(np.float32)

            data      = {'iEEG': signal, 'label_time': label_time}
            file_name = hosp_id + '_' + subject_id + '_' + PE_id + '_E' + str(nfile)

            df_aux = {  
                        'rns_id': hosp_id + '-' + subject_id, 
                        'data'  : file_name,
                        'label' : labels[nfile],
                        'time'  : times[nfile]
                     }
            
            df = pd.concat([df, pd.DataFrame([df_aux])], ignore_index=True)

            np.save(OUTDATA_DIR + file_name, data)

    df.to_csv(OUTMETADATA_DIR + 'allfiles_metadata.csv', index=False)