# -*- coding: utf-8 -*-
"""
Created on nov 4, 2022

This function makes the final annotations
The 24 models outputs will be processed and analyzed to make a final prediction
weighted average ensamble is used in outouts probs
@author: vpeterson
"""

# check predictions
import numpy as np 
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'utilities')))
import IO  
import Epochs 
import torch 
from numpy.linalg import norm 
import pandas as pd
from numpy import average

#%% define custom functions

def compute_wae(outputs, weights):
    # post process the output given the threshold
    predicted_av = average(outputs, axis = 0, weights = weights)
    
    y_torch = torch.from_numpy(predicted_av)
    n = torch.nn.Threshold(thr, 0)
    y_predicted = n(y_torch)
    l_predicted = np.zeros((len(y_predicted,)))
    idx_predicted = np.where(y_predicted.sum(dim=1)>0.0)[0].tolist()
    l_predicted[idx_predicted] = 1  
    t_predicted_samples = np.zeros((len(y_predicted,)))
    t_predicted_samples[idx_predicted] = y_predicted[idx_predicted].argmax(dim=1).numpy()
    t_predicted_samples = t_predicted_samples.astype(int)
    
    ECOG_SAMPLE_RATE =250
    TT = 1000 # window length
    overlap = 500 #
    win_len = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    hop_len = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000)   # Length of hop between windows.
    fs=250
    shape = 22500
    time = np.arange(win_len/2, shape + win_len/2 + 1,
                         win_len - (win_len-hop_len))/float(fs)
    time -= (win_len/2) / float(fs)
    
    t_predicted = time[t_predicted_samples]
    return l_predicted, t_predicted, y_predicted

def create_annot_file(address, file_name, estimated_label, estimated_time, offset_time, annot_eof, rec_len):
    twindow = 90.0
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]
    idx_longtrials = np.where(np.asarray(rec_len) > 91)[0]
    
    if annot_eof.size == 1:
        annot_eof = np.expand_dims(annot_eof, axis=0)
    
    f = open(address + file_name + "_EOF_AI.txt","w+")  
    f.write("Onset" + "," + "Annotation" + "\n")
    cont=0
    for i in range(len(annot_eof)):
        # print('cont = %s, iter = %s' % (str(cont) , str(i)))
        len_archive = rec_len[i]
        # short files
        if (i==idx_shorttrials).any():
            f.write("{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")
            continue
       
        # long files
        if (i==idx_longtrials).any():
            nwindows = round(len_archive/twindow)
            for nw in range(nwindows):
                if nw ==0:
                    if estimated_label[cont]==0:
                        #f.write( "{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")   
                        continue
                    else:
                        time_onset = estimated_time[cont]+ offset_time[cont]
                        if i!=0:
                            if not (annot_eof[i-1] <= time_onset <= annot_eof[i]):
                                raise TypeError("sz_on not in between two eof")

                        else:
                            if not (time_onset < annot_eof[i]):
                                raise TypeError("sz_on not in between two eof")
                        f.write("{:.3f}".format(time_onset) + "," + "sz_on" + "\n")
                    #f.write( "{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")   

                else:
                    cont +=1

                    if estimated_label[cont]==0:
                        #f.write( "{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")   

                        continue
                    else:
                        time_onset = estimated_time[cont]+ offset_time[cont]
                        if i!=0:
                            if not (annot_eof[i-1] <= time_onset <= annot_eof[i]):
                                if  (annot_eof[i-1] <= time_onset - 1 <= annot_eof[i]):
                                    time_onset = time_onset- 1
                                else:
                                    raise TypeError("sz_on not in between two eof")

                        else:
                            if not (time_onset < annot_eof[i]):
                                raise TypeError("sz_on not in between two eof")
                        f.write("{:.3f}".format(time_onset) + "," + "sz_on" + "\n")
            f.write( "{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")   
        else:
             # regular files
            if estimated_label[cont]==0:
                f.write("{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")
            else:
                time_onset = estimated_time[cont]+ offset_time[cont]
                if i!=0:
                    if not (annot_eof[i-1] <= time_onset <= annot_eof[i]):
                        raise TypeError("sz_on not in between two eof")
    
                else:
                    if not (time_onset < annot_eof[i]):
                        raise TypeError("sz_on not in between two eof")
                f.write("{:.3f}".format(time_onset) + "," + "sz_on" + "\n")
                f.write( "{:.3f}".format(annot_eof[i]) + "," + "eof" + "\n")
        
        if not (i==idx_shorttrials).any() or len(idx_shorttrials)==0:
            cont+=1
    if cont != len(estimated_time):
        raise TypeError("cont not equal to number of epochs")
    f.close()
#%%
FREQ_MASK_PARAM = 10
TIME_MASK_PARAN = 20
N_CLASSES = 1
learning_rate = 1e-3
batch_size = 128
epochs = 20
num_workers = 4
hparams = {
          "n_cnn_layers": 3,
            "n_rnn_layers": 3,
            "rnn_dim": [150, 100, 50],
            "n_class": N_CLASSES,
            "out_ch": [8,8,16],
            "dropout": 0.3,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "epochs": epochs
    }



DATA_DIR = "X:/RNS_DataBank/MGH/MGH/"
SPE_DIR = 'C:/Users/vp820/Documents/ESPnet/TimeLabelZeropadAll/MGH/'
meta_data_file = 'C:/Users/vp820/Documents/ESPnet/TimeLabelZeropadAll/MGH/METADATA/allfiles_metadata.csv'
pit_subject_info = 'X:/iESPnet/Data/Metadatafiles/subjects_info_zeropadall_nothalamus.csv'
mgh_subject_info = 'X:/iESPnet/Data/Metadatafiles/subjects_info_zeropadall_mgh.csv'

# this is the path where PITT LOSO results are saved. Used to calculate the weigths
save_path = 'X:/iESPnet/Models/PITT/'
# this is the path where the outputs for MGH data where saved (one per PITT model)
output_path = 'X:/iESPnet/AnnotMGH/outputs/'
# this is the pat where precitions are saved
prediction_path = 'X:/RNS_DataBank/MGH/MGH/'

TRAINING = False
if TRAINING:
    type_prediction = 'prediction_tr'
else:
    type_prediction = 'prediction_te'
    
#%% get subject list
RNSIDS_MGH = IO.get_subfolders(DATA_DIR)
pit_df_subjects = pd.read_csv(pit_subject_info)
mgh_df_subjects = pd.read_csv(mgh_subject_info)  
# keep annoted patients
RNSids_all_MGH = 'MGH-' + mgh_df_subjects.rns_id_np
idx_match = [idx for idx, s in enumerate(RNSids_all_MGH) if s in RNSIDS_MGH ] # ask for a and b in name
RNSIDS_MGH_b = [idd + 'b' for idd in RNSIDS_MGH]
idx_match_b = [idx for idx, s in enumerate(RNSids_all_MGH) if s in RNSIDS_MGH_b] # ask for a and b in name
idx_ = list(set(idx_match).union(set(idx_match_b)))

mgh_df_subjects_match = mgh_df_subjects.iloc[idx_,:]
# clean thalamus group
idx_nonthalamus =  [idx for idx, s in enumerate(mgh_df_subjects_match.group) if "thalamus" not in s ]
mgh_df_subjects_match_nothalamus = mgh_df_subjects_match.iloc[idx_nonthalamus,:]
RNSIDS_MGH = ('MGH-' + mgh_df_subjects_match_nothalamus.rns_id_np).tolist()
# remove b
RNSIDS_MGH = [ss.strip('b') for ss in RNSIDS_MGH] 

RNSIDS_PIT=pit_df_subjects.rns_deid_id
thr = 0.2
#%%
len(RNSIDS_MGH)

for s in range(len(RNSIDS_MGH)):
    print('Running testing for subject ' + RNSIDS_MGH[s] + ' [s]: ' + str(s))
   
    df = pd.read_csv(meta_data_file) 

    test_df = df[df.rns_id==RNSIDS_MGH[s]]
    test_df.reset_index(drop=True, inplace=True)
    
    # where the PITT results are for every MGH patient
    saved_predictions = output_path + RNSIDS_MGH[s]+'/results/'
    
    # where the annot files are willing to be saved
    save_predictions = prediction_path + RNSIDS_MGH[s] + '/' + 'iEEG/'
    
        
    test_df['epoch'] = test_df.data.apply(lambda x: x.split("_")[2])
    test_epochs = test_df.epoch.unique()
    
    annot_files = IO.get_files(DATA_DIR, RNSIDS_MGH[s], endswith='.txt',
                                     Verbose=False)
    data_files = IO.get_files(DATA_DIR, RNSIDS_MGH[s], Verbose=False)

    for nfile, epoch_counter in enumerate(test_epochs):
        annot_file = [f for f in annot_files if epoch_counter in f]
        data_file = [f for f in data_files if epoch_counter in f]

        annot_file = annot_file[0]
        data_file = data_file[0]
        X, offset_time = Epochs.get_fixed_epochs_zeropad(data_file, annot_file)
        # it could happen that all trials are short, and then no epochs have
        # been extracted.
        if len(X) == 0:
            continue
        
        # EOF
        annot_eof = np.loadtxt(annot_file, delimiter=',',
                        dtype=float)
        
        eof_event_time = np.hstack((0.0, annot_eof))
        # # check recording lenght is at least 90s
        rec_len = [t - s for s, t in zip(eof_event_time, eof_event_time[1:])]
        idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]
        idx_longtrials = np.where(np.asarray(rec_len) > 91)[0]

        
        

        # define a test_df_pe
        # the programming epoch information is in test_df.data
        # "data" column is the file name for the spectrogram
        test_df_pe = test_df[test_df['epoch'] == epoch_counter]
        
        # get time offset
        file_name = 'MGH' + '-' + RNSIDS_MGH[s][4:] + '_PE' + epoch_counter
        session_path = SPE_DIR + file_name
        
        labels = []
        times = []
        output = []
        # make the prediction for every PIT model
        weights = []
        for ss in range(len(RNSIDS_PIT)):
            
            results = np.load(saved_predictions+ RNSIDS_MGH[s]+ 'results_pe_'+ str(epoch_counter) + '_model_'+ str(ss) + '.npy', allow_pickle=True).tolist()
            predictions = results['prediction_te']
            y_prob = predictions['y_prob']
            
            output.append(y_prob)
            
            # get results to compute weigth
            save_predictions_pitt = save_path + RNSIDS_PIT[ss]+'/results/'   
            results =  np.load(save_predictions_pitt+ RNSIDS_PIT[ss]+ 'results.npy', allow_pickle = True)
            results = results.tolist()
            
            
            prediction_te = results[type_prediction]
            subject_df = pit_df_subjects[pit_df_subjects.rns_deid_id==RNSIDS_PIT[ss]]
            q = subject_df.Nsz/subject_df.Nfiles
            
            p = (subject_df.Nfiles-subject_df.Nsz)/subject_df.Nfiles
            F1_coin = np.array(2*q/(q+1))
            
            we = (prediction_te['f1']-F1_coin)/F1_coin/100
            weights.append(we)
        
        weights_n = weights/norm(weights,1)
        weights = np.squeeze(np.array(weights))
        weights[weights<0.0]=0.0
        outputs = np.array(output)    
            

        estimated_label, estimated_time, estimated_output = compute_wae(outputs, weights)  #weighted average ensemble
        # create TXT annot file
        create_annot_file(save_predictions, file_name, estimated_label, estimated_time, offset_time, annot_eof, rec_len)
        # save np to subsequent analysis
        predict_ = { "y_pred": estimated_output,
                    "l_pred": estimated_label,
                    "t_pred": estimated_time,

                    }
        np.save(saved_predictions + '/'+ file_name + '_ensemble_results.npy', predict_)
        


        
            