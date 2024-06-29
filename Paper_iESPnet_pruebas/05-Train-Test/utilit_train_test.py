
import torch
import numpy as np
import os 
import mne
from scipy import fft as sp_fft
import torch
import torchaudio.transforms as T

def make_weights_for_balanced_classes(train_df, classes, n_concat=2):
    class_sample_count = np.zeros(len(classes,), dtype=int)
    for n, cl in enumerate(classes):
        class_sample_count[n] = sum(train_df.label==cl)
       
    weights = (1 / class_sample_count)
    target = train_df.label.to_numpy()
    samples_weight = weights[target]
    
    for i in range(n_concat):
        if i == 0:
            sampler = samples_weight
        else:
            sampler = np.hstack((sampler, samples_weight))
    
    return torch.tensor(sampler , dtype=torch.float)

# funciones para cargar los datos 

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

def get_epochs_zeropad_all(data_file, events, twindow=90.0):
    """
    Extract epochs from file at a given twindow length.

    This function segments data in fixed epochs. If a file is shorter than 
    90 s, then zeropadding. If it is longer, then non-overlapping 90 s segments
    are extracted. 
    
    Useful for creating spectrogram of edf files with annotations (aka PITT)


    Parameters
    ----------
    data : array, shape(n_channels, n_samples)
        either cortex of subcortex data to be epoched.
    events : array, shape(n_events,2)
        All events that were found by the function
        'get_events'.
        The first column contains the event time seconds and the second column
        contains the event id.
    twindow : float
        length of the epoch.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    Y : array, shape(n_events, n_samples)
        label information, no_sz=0, sz_on=1, sz=2.
        See 'get_events' for more information regarding labels names.
    Z : array, shape(n_events, n_samples)
        time of the sz_on.
    """
    raw = mne.io.read_raw_edf(data_file)
    sf = raw.info['sfreq']
    data = raw.get_data()
    n_channels, time_rec = np.shape(data)
    # labels
    idx_eof = np.where(events[:, 1] == 0)[0]
    eof_event_time = events[idx_eof, 0]  # in seconds
    # add zero for first eof
    eof_event_time = np.hstack((0.0, eof_event_time))
    # # check recording lenght is at least 90s
    rec_len = [t - s for s, t in zip(eof_event_time, eof_event_time[1:])]
    # files shorter than 60 are discarded.
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]

    # delete short trials:
    idx_eof_new = np.delete(idx_eof, idx_shorttrials)
    eof_event_time_new = events[idx_eof_new, 0]
    rec_len_new = np.delete(rec_len, idx_shorttrials)

    # check if annot time is not longer than file time
    idx_stop = np.where(np.dot(eof_event_time_new, sf) > time_rec)

    # avoid trying to read those trials
    idx_eof_ = np.delete(idx_eof_new, idx_stop)
    nTrials = len(idx_eof_)

    Y = np.zeros((nTrials,)).astype(int)  # labels
    Z = np.zeros((nTrials,))  # time
    n_samples = int(round((twindow)*sf))
    X = np.zeros((nTrials, n_channels, n_samples))
    
    # extract epochs starts
    for i in range(nTrials):
        len_archive = rec_len_new[i]
        time_sof = eof_event_time_new[i] - len_archive
        start_epoch = int(round(time_sof * sf))
        if len_archive >= twindow:
            stop_epoch = int(round((time_sof + twindow) * sf))
            final_epoch = stop_epoch
            epoch = data[:, start_epoch:stop_epoch]
        else:
            stop_epoch = int(round((time_sof + len_archive) * sf))
            final_epoch = int(round((time_sof + twindow) * sf))
            epoch = data[:, start_epoch:stop_epoch]
            # pad the time that is not part of the shortened epoch to reach twindow
            time_to_pad = final_epoch - stop_epoch # in samples
        
            
            if time_to_pad > 0:
             
                # using zeros
                pad_data = np.zeros((4, time_to_pad))
                epoch = np.concatenate((epoch,pad_data),1)
                # now extract only the data fit to the frame
                epoch = epoch[:, :n_samples]
            # using 0.5s to pad.
            pad_length = int(round(0.1 * sf))
            pad_data = data[:,(stop_epoch - pad_length):stop_epoch]
            num_pad_reps = int(np.floor(time_to_pad/pad_length) + 1)
            total_pad_data = np.tile(pad_data,num_pad_reps)
            epoch = np.concatenate((epoch,total_pad_data),1)
            # now extract only the data fit to the frame
            epoch = epoch[:, :n_samples]
        # check lenght epoch
        if (final_epoch - start_epoch) != n_samples:
            raise ValueError('epoch lengths does not represent time window')
        
        X[i] = epoch

        if idx_eof_[i] != len(events)-1:
            if events[idx_eof_[i] - 1, 1] != 0 and (idx_eof_[i] - 1) != -1:  # then is sz_on
                # label
                Y[i] = events[idx_eof_[i] - 1, 1]
                # time
                time_sz = events[idx_eof_[i] - 1, 0]
                time = time_sz - time_sof
                time_eof = events[idx_eof_[i], 0]
                if len_archive > twindow:  # large archive
                    if time > twindow:  # sz happens and epoch didn't capture it
                        n_90 = int(len_archive / twindow)  # n of 90 s in axive
                        t_90 = time / twindow  # in which n time happens
                        if n_90 < t_90:  # sz is happening closer the end
                            # redefine epoch from eof to - 90
                            start_epoch = int(round((time_eof - twindow) * sf))
                            stop_epoch = int(round(time_eof * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            X[i] = epoch

                            # time
                            time = twindow - (time_eof - time_sz)
                        else:
                            # make sure we pick up the correct 90 s
                            start_epoch = int(round((time_sof + int(t_90)*twindow) * sf))
                            stop_epoch = int(round((time_sof + int(t_90)*twindow + twindow) * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            X[i] = epoch

                            # time
                            time = time_sz - time_sof - int(t_90)*twindow 
                if np.sign(time) == -1:
                    raise ValueError('Time cannot be negative')
                Z[i] = time
    return X, Y, Z

def get_events(annot_file, Verbose=False):
    """
    Clean annotation file and generate event array.

    Parameters
    ----------
    annot_files : str
        annot_file.

    Returns
    -------
    events : array
        time and event type information.

    """

    annot_ = np.loadtxt(annot_file, delimiter=',', skiprows=1, dtype=str)
    annot_ = annot_[:,[0,-1]]
    # we define here the valid labels
    idx = [idx for idx, s in enumerate(annot_[:, 1]) if "eof" not in s]
    idx2 = [idx for idx, s in enumerate(annot_[idx, 1]) if "sz_" not in s]

    idx_2delete = []
    for i in range(len(idx2)):
        idx_2delete.append(idx[idx2[i]])

    idx_2delete = np.asarray(idx_2delete).astype(int)
    if Verbose:
        print(str(len(idx_2delete))+' events deleted')

    events = np.delete(annot_, idx_2delete, 0)
    # change and arrange categorical to discrete labels
    # eof == 0, sz_on_*==1, sz*==2
    idx_eof = [idx for idx, s in enumerate(events[:, 1]) if "eof" in s]
    idx_sz_ = [idx for idx, s in enumerate(events[:, 1]) if "sz" in s]
    idx_sz_on = [idx for idx, s in enumerate(events[:, 1]) if "sz_on" in s]
    # the sz_ get everything with sz. the class "sz with onset" will be get
    # as a difference between the idx_sz and idx_sz_on
    idx_sz = set(idx_sz_) - set(idx_sz_on)
    idx_sz = list(idx_sz)

    events[idx_eof, 1] = 0
    events[idx_sz_on, 1] = 1
    events[idx_sz, 1] = 2

    events = events.astype(float)
    return events

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