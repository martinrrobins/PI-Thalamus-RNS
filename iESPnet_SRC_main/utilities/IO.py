#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
useful functions for navigating through the folders and files
@author: vpeterson
"""
import os
import numpy as np
import librosa
import torch
import torchaudio.transforms as T
from scipy import fft as sp_fft


def get_subfolders(subject_path, Verbose=False):
    """
    Get subfolder from a given path. Useful for getting all patients list.

    given an address, provides all the subfolder included in such path.

    Parameters
    ----------
    subject_path : string
        address to the subject folder
    Verbose : boolean, optional

    Returns
    -------
    subfolders : list
    """
    subfolders = []
    for entry in os.listdir(subject_path):
        if os.path.isdir(os.path.join(subject_path, entry)):
            subfolders.append(entry)
            if Verbose:
                print(entry)

    return subfolders


def get_data_files(address, subfolder, endswith='.edf', Verbose=True):
    """
    Get file from a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.

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
    files = []
    session_path = address + '/' + subfolder 
    # only data whose annotation exist will be loaded
    annot_files = get_annot_files(address, subfolder, Verbose=False)
    if endswith == '.edf':
        prestr = len(endswith)+1  # for checking pre-file extention string
        for f_name in os.listdir(session_path):
            if f_name.lower().endswith(endswith) and f_name[-prestr].isnumeric():
                aux = f_name[:-len(endswith)]  # check annot
                if any(aux in s for s in annot_files):
                    files.append(session_path + '/' + f_name)
                    if Verbose:
                        print(f_name)
    else:
        for f_name in os.listdir(session_path):
            if f_name.endswith(endswith):
                aux = f_name[:-(len(endswith)+5)]  # check annot
                if any(aux in s for s in annot_files):
                    files.append(session_path + '/' + f_name)
                    if Verbose:
                        print(f_name)
    files.sort()
    return files

def get_files(address, subfolder, endswith='.edf', Verbose=True):
    """
    Get file from a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.

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
    files = []
    session_path = address + '/' + subfolder+ '/iEEG'
   
    if endswith == '.edf':
        prestr = len(endswith)+1  # for checking pre-file extention string
        for f_name in os.listdir(session_path):
            if f_name.lower().endswith(endswith) and f_name[-prestr].isnumeric():
                files.append(session_path + '/' + f_name)
                if Verbose:
                    print(f_name)
    else:
        for f_name in os.listdir(session_path):
            if f_name.lower().endswith(endswith):
                files.append(session_path + '/' + f_name)
                if Verbose:
                    print(f_name)
    files.sort()
    return files


def get_annot_IA_files(address, subfolder):

    """
    Get prediction files done by the net for a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.

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
    files = []
    session_path = address + '/' + subfolder + '/iEEG'
    annot_eof = 'sz-ai.txt'
    files = [session_path +'/'+ f for f in os.listdir(session_path) if f.lower().endswith(annot_eof)]
    files.sort()
    return files


def get_annot_files(address, subfolder, Verbose=True):
    """
    Get annot file from a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.
    To get info from vhdr_file please see 'get_sess_run_subject'

    Parameters
    ----------
    subject_path : string
    subfolder : list
    Verbose : boolean, optional

    Returns
    -------
    iles : list
        list of addrress to access to a particular vhdr_file.
    """
    files = []
    session_path = address + '/' + subfolder 
    annot1_eof = 'sz-vk.txt'
    anno2_eof = 'sz-nz.txt'  # this files contain both VK and NZ annots
    txt_files_1 = [f for f in os.listdir(session_path) if f.lower().endswith(annot1_eof)]
    txt_files_2 = [f for f in os.listdir(session_path) if f.lower().endswith(anno2_eof)]

    # anno2_eof files are annotation files of first order priority
    aux_prefix = np.repeat(session_path + '/', len(txt_files_2))
    files = [x + y for x, y in zip(aux_prefix, txt_files_2)]
    # now check if annot1 exists but not annot2
    for f, f_name in enumerate(txt_files_1):
        aux = f_name[:-len(annot1_eof)]  # aux name file which annot
        if any(aux in s for s in txt_files_1) and not any(aux in s for s in txt_files_2):
            files.append(session_path + '/' + f_name)
        if Verbose:
            print(f_name)
    # check files are not empty files
    files = [ f for f in files if  os.stat(f).st_size != 0]
    files.sort()
    return files


def get_patient_PE(data_file, magicword_1):
    """
    Given a the data_file string return the subject, and PE.
    magicword refers to the string from which the patient folder starts

    Parameters
    ----------
        data_file (string): [description]

    Returns
    -------
        hops, subject, PE
    """
    magicword = magicword_1 + '/'
    hosp = data_file[data_file.find(magicword) + len(magicword):
                     data_file.find(magicword) + len(magicword) + 3]
    to_find = magicword + hosp + '-'

    subject = data_file[data_file.find(to_find)+len(to_find):
                        data_file.find(to_find)+7+len(to_find)]

    str_PE = data_file[data_file.find('PE'):]
    PE = str_PE[2:-4]

    return hosp, subject, PE

def get_spectrogram_2(signal, device, fs, n_fft = 256, win_len = None, hop_len = None, top_db = 40.0, power = 2.0):

    np.random.seed(0)

    wind_dic = {'periodic': True, 'beta': 10}

    spectrogram = T.Spectrogram(
                                n_fft=n_fft, 
                                win_length=win_len,
                                hop_length=hop_len, 
                                pad=0,
                                window_fn =torch.kaiser_window,
                                normalized=False,
                                power=power, 
                                wkwargs=wind_dic
                               )
    
    #spectrogram = spectrogram.to(device)
    
    time   = np.arange(win_len/2, signal.shape[-1] + win_len/2 + 1, win_len - (win_len-hop_len))/float(fs)
    time  -= (win_len/2) / float(fs)

    freqs  = sp_fft.rfftfreq(n_fft, 1/fs)

    spec   = spectrogram(signal)
    spec   = spec.detach().cpu().numpy()
    spec   = librosa.power_to_db(spec, top_db=top_db)

    # save up to 60 Hz
    idx_60 = np.where(freqs <= 60)[0][-1]

    spec   = spec[:, :, :idx_60, :]

    return spec #, time, freqs  ver de agregar para plot 

