#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate metadatafile with files annotation information
@author: vpeterson
"""
import pandas as pd
import os
import numpy as np
import IO # customized functions for navigating throught the folders and files
#%%
def write_df_data_files(address, subfolder, annot_files, endswith='.EDF'):
    """
    Get file from a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.
    To get info from vhdr_file please see 'get_sess_run_subject'

    Parameters
    ----------
    subject_path : string
    subfolder : list
    endswith ; string
    Verbose : boolean, optional

    Returns
    -------
    iles : list
        list of addrress to access to a particular vhdr_file.
    """
    session_path = address + '/' + subfolder + '/iEEG'
    prestr = len(endswith)+1
    files_ = [f for f in os.listdir(session_path) if f.endswith(endswith) and
              f[-prestr].isnumeric()]
    subject_id = np.repeat(subfolder, len(files_))

    first_annot = []
    second_annot = []
    Nsz_on = []
    Nsz = []
    Nnz1 = []
    Nnz5 = []

    # for checking pre-file extention string
    for f, f_name in enumerate(files_):
        aux = f_name[:-len(endswith)]  # check annot
        if any(aux in s for s in annot_files):
            # read annot
            idx = [(i, s.index(aux))  for i, s in enumerate(annot_files)
                   if aux in s][0][0]
            if annot_files[idx][-prestr] == 'K':
                second_annot.append('NO')
            else:
                second_annot.append('YES')
            # if file not empty
            if not os.stat(annot_files[idx]).st_size == 0:
                first_annot.append('YES')
                annot_ = np.loadtxt(annot_files[idx], delimiter=',', skiprows=1, dtype=str)
                Nsz_on.append(sum(annot_[:,1]=='sz_on') +
                              sum(annot_[:,1]=='sz_on_l') +
                              sum(annot_[:,1]=='sz_on_r'))
                Nsz.append(sum(annot_[:,1]=='sz')+
                              sum(annot_[:,1]=='sz_l') +
                              sum(annot_[:,1]=='sz_r'))
                Nnz1.append(sum(annot_[:,1]=='NZ1'))
                Nnz5.append(sum(annot_[:,1]=='NZ5'))
            else:
                first_annot.append('EMPTY')
                Nsz_on.append(sum(annot_[:,1]==' '))
                Nsz.append(sum(annot_[:,1]==' '))
                Nnz1.append(sum(annot_[:,1]==' '))
                Nnz5.append(sum(annot_[:,1]==' '))
        else:
            first_annot.append('NO')
            second_annot.append('NO')
            Nsz_on.append(sum(annot_[:,1]==' '))
            Nsz.append(sum(annot_[:,1]==' '))
            Nnz1.append(sum(annot_[:,1]==' '))
            Nnz5.append(sum(annot_[:,1]==' '))


    data = {'ID': subject_id,
            'File': files_,
            'VK annot file': first_annot,
            'NZ annot file': second_annot,
            '#sz_on': Nsz_on,
            "#sz": Nsz,
            '#Nnz1': Nnz1,
            '#Nnz5':Nnz5}
    col_names = ['ID', 'File', 'VK annot file', 'NZ annot file',
                 '#sz_on', "#sz", '#Nnz1', '#Nnz5']
    df =  pd.DataFrame(data, columns=col_names)

    return df
#%%
DATA_DIR = "/mnt/Nexus2/RNS_DataBank/PITT/"
df_subjects = pd.read_csv('/mnt/Nexus2/ESPnet/Data/Metadatafiles/subjects_info_nothalamus.csv')

RNSIDS=df_subjects.rns_deid_id
# RNSIDS = IO.get_subfolders(DATA_DIR)

df_all = pd.DataFrame()
for s in range(len(RNSIDS)):
    if RNSIDS[s] != 'PIT-RNS1090':  # this subject's data is useless
        continue
    annot_files = IO.get_annot_files(DATA_DIR, RNSIDS[s], endswith='.TXT',
                                     Verbose=False)
    df_s = write_df_data_files(DATA_DIR, RNSIDS[s], annot_files)
    df_all = pd.concat([df_all, df_s])

total_sz = sum(df_all['#sz_on']) + sum(df_all['#sz'])
total_sz_on = sum(df_all['#sz_on'])

disag_sz = sum(df_all['#Nnz5'])
added_sz = sum(df_all['#Nnz1'])

print('The TOTAL number of marked sz is: ' + str(total_sz))
print('The TOTAL number of marked sz_on is: ' + str(total_sz_on))

print('The TOTAL number of disagreed sz is: ' + str(disag_sz))
print('The TOTAL number of added sz is: ' + str(added_sz))

# save DF
# df_all.to_csv('files_metadata_annots_nothalamus.csv', index=False)
