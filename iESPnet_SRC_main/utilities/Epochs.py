#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Several functions that help to get data structure in epochs
@author: vpeterson, Sep 2021, Updated @Dec 2022.
"""
import numpy as np
import mne

def get_fixed_epochs_zeropad(data_file, annot_file, twindow=90.0):
    """
    Extract epochs from file at a given twindow length.

    This function segments data in fixed epochs. If a file is shorter than 
    90 s, then zeropadding. If it is longer, then non-overlapping 90 s segments
    are extracted. 
    
    Note that no information regarding seizure onset is used.
    Useful to create spectrogram of edf files without annotations (aka MGH)
    

    Parameters
    ----------
    data_file : str, 
    annot_file : str, shape(n_events,2)
        
    twindow : float
        length of the epoch.
   
    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    T : array, shape(n_events, n_samples)
        time at wich an file started. Useful for tracking segment time info
    """
    raw = mne.io.read_raw_edf(data_file)
    sf = raw.info['sfreq']
    data = raw.get_data()
    n_channels, time_rec = np.shape(data)
    # labels
    annot_ = np.loadtxt(annot_file, delimiter=',',
                        dtype=float)

    eof_event_time = annot_ # in seconds
    # add zero for first eof
    eof_event_time_0 = np.hstack((0.0, eof_event_time))
    # # check recording lenght is at least 90s
    rec_len = [t - s for s, t in zip(eof_event_time_0, eof_event_time_0[1:])]
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]

    # check if annot time is not longer than file time
    idx_stop = np.where(np.dot(eof_event_time, sf) > time_rec)


    if eof_event_time.size == 1:
        eof_event_time = np.expand_dims(eof_event_time, axis=0)
    nTrials = len(eof_event_time)

    n_samples = int(round((twindow)*sf))
    X = []
    T = [] # this is for saving the recording time
    if len(eof_event_time) -  len(idx_shorttrials)!= 0:
            
        
        # extract epochs starts
        for i in range(nTrials):
            if (i==idx_shorttrials).any():
                # print('skipped epoch')
                continue
            len_archive = rec_len[i]
            time_sof = eof_event_time[i] - len_archive
                       
            start_file = int(round(time_sof * sf))
            stop_file = int(round((time_sof + len_archive) * sf))
            data_e = data[:, start_file:stop_file]
            
            time_offset = time_sof
    
            if len_archive >= twindow:
                # check how many non-overlapping windows can be extracted
                nwindows = round(len_archive/twindow)
                for nw in range(nwindows):
                    if nw ==0:
                        start_epoch = 0
                        stop_epoch = int(twindow* sf)     
                    else:
                        start_epoch = stop_epoch
                        stop_epoch = start_epoch + int(twindow* sf)
                        time_offset = time_sof + twindow
                        
                    epoch = data_e[:, start_epoch:stop_epoch]
                    
                    time_to_pad = int(twindow* sf) - len(epoch.T) # in samples
    
                    
                    if time_to_pad > 0:
                     
                        # using zeros
                        pad_data = np.zeros((4, time_to_pad))
                        epoch = np.concatenate((epoch,pad_data),1)
                        # now extract only the data fit to the frame
                        epoch = epoch[:, :n_samples]
                    
                    X.append(epoch)
                    T.append(time_offset)
            else:
                start_epoch = 0
                stop_epoch = int(twindow* sf) 
                epoch = data_e[:, start_epoch:stop_epoch]
                
                # pad the time that is not part of the shortened epoch to reach twindow
                time_to_pad = int(twindow* sf) - len(epoch.T) # in samples
                if time_to_pad > 0:
    
                    # using zeros
                    pad_data = np.zeros((4, time_to_pad))
                    epoch = np.concatenate((epoch,pad_data),1)
                    # now extract only the data fit to the frame
                    epoch = epoch[:, :n_samples]
                
                X.append(epoch)
                T.append(time_offset)
            # check lenght epoch
            if np.size(epoch, axis=1) != n_samples:
                raise ValueError('epoch lengths does not represent time window')


        X = np.dstack(X).T
        X = np.swapaxes(X, 2, 1)

    return X, T



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

def get_segments_events(events, twindow=90.0):
    """
   
    Returns
    -------
   
    Y : array, shape(n_events, n_samples)
        label information, no_sz=0, sz_on=1, sz=2.
        See 'get_events' for more information regarding labels names.
    T : array, shape(n_events, n_samples)
        time of the sz_on.
    """
    idx_eof = np.where(events[:, 1] == 0)[0]
    eof_event_time_ = events[idx_eof, 0]  # in seconds
    # add zero for first eof
    eof_event_time = np.hstack((0.0, eof_event_time_))
    # # check recording lenght is at least 90s
    rec_len = [t - s for s, t in zip(eof_event_time, eof_event_time[1:])]
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]

    sf = 250
    # rec_len = np.delete(rec_len, idx_shorttrials)
    if eof_event_time.size == 1:
        eof_event_time = np.expand_dims(eof_event_time, axis=0)
    nTrials = len(eof_event_time_)

    Y = []
    T = [] # this is for saving the recording time
    if len(eof_event_time) -  len(idx_shorttrials)!= 0:
        # extract epochs starts
        for i in range(nTrials):
            
            # skip short trials
            if (i==idx_shorttrials).any():
                # print('skipped epoch')
                continue
            len_archive = rec_len[i]
            time_sof = eof_event_time_[i] - len_archive # time start of file
            #label 
            Y.append(int(events[idx_eof[i] - 1, 1]))
            if events[idx_eof[i] - 1, 1] != 0 and (idx_eof[i] - 1) != -1:  # then is sz_on               
                # time
                time_sz = events[idx_eof[i] - 1, 0]
                time = time_sz - time_sof
                T.append(time)
            else:
                T.append(0.0)
       

    return np.asarray(Y), np.asarray(T)