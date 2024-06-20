#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the model

@author: vpeterson
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'utilities')))
import IO
#%%
DATA_DIR = "/YOURPATH/RNS_DataBank/PITT/" #change these paths
SPE_DIR = '/YOURPATH/iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll/PITT/'
# get metadata file
meta_data_file = 'YOURPATH/iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll/PITT/METADATA/allfiles_metadata.csv'
df = pd.read_csv(meta_data_file) 


df_subjects = pd.read_csv('/YOURPATH/iESPnet/Data/Metadatafiles/subjects_info_nothalamus.csv')

RNSIDS=df_subjects.rns_deid_id
RNSIDS_all = IO.get_subfolders(DATA_DIR)
# eliminate class 2
df.drop(df[df['label'] == 2].index, inplace = True)
df.head()
# eliminate thalamus patients from list
for ii in range(len(RNSIDS_all)):
    if not any(RNSIDS.str.contains(RNSIDS_all[ii])):
        df.drop(df[df['rns_id'] == RNSIDS_all[ii]].index, inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv('/YOURPATH/iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll/PITT/METADATA/allfiles_nothalamus_metadata.csv', index=False)
