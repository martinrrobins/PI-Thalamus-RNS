#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate metadatafile with subject info information
@author: vpeterson
"""
import pandas as pd
import os
import numpy as np
import sys
sys.path.insert(1, '../utilities') 
import IO # customized functions for navigating throught the folders and files
#%%
# read RNS patient info
path_file = 'X:/iESPnet/Data/Metadatafilesn/RNSpatients_info.csv' # saved in Nexus2
file_subjects =  pd.read_csv(path_file)
# read metada file
meta_data_file = 'YOURPATH/allfiles_metadata.csv' # if non thalamos should be allfiles_nothalamus_metadata

# delete files at which there was an iESP but without sz onset (label==2)
df =  pd.read_csv(meta_data_file)
df.drop(df[df['label'] == 2].index, inplace = True)   

#%%
DATA_DIR = "X:/RNS_DataBank/PITT/" #change this address if needed
RNSIDS = IO.get_subfolders(DATA_DIR)

n_total = []
n_sz = []
for s in range(len(RNSIDS)):
    subject_df = df[df.rns_id==RNSIDS[s]]
    n_total.append(len(subject_df.label))
    n_sz.append(sum(subject_df.label))


file_subjects['Nfiles'] = n_total #number of spectrograms
file_subjects['Nsz'] = n_sz   # from the total how many have seizure
# save DF
file_subjects.to_csv('subjects_info_nopadall.csv', index=False) #change this name if needed
