# -*- coding: utf-8 -*-
"""
Created on Jan 19, 2022

This function makes prediction on MGH data based on the individuals PITT models
24 output models will be the results of this script
The final decision labelling will be made based on model bagging
@author: vpeterson
"""

# check predictions
import numpy as np 
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, mean_absolute_error, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'utilities')))

import IO     
from Generator import SeizureDatasetLabelTime, StatsRecorder, normalize_spec, scale_spec, permute_spec, smoothing_label
from Model import ESPfullnet
import IO
import pandas as pd
from TrainEval import test_model, train_model, get_thr_output, get_performance_indices

# this is the path where the PITT models are
save_path = 'X:/iESPnet/Models/PITT/' 
# this is the path where each model predictions will be saved
output_path = 'X:/iESPnet/AnnotMGH/outputs/'

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
# Remember to make a local copy of the spectrograms to have this code run fast
SPE_DIR = 'X:/iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll/MGH/'
meta_data_file = 'C:/Users/vp820/Documents/ESPnet/TimeLabelZeropadAll/MGH/METADATA/allfiles_metadata.csv'


TRAINING = False
if TRAINING:
    type_prediction = 'prediction_tr'
else:
    type_prediction = 'prediction_te'
    
# get subject list
RNSIDS_MGH = IO.get_subfolders(DATA_DIR)
pit_subject_info = 'X:/iESPnet/Data/Metadatafiles/subjects_info_zeropadall_nothalamus.csv'
pit_df_subjects = pd.read_csv(pit_subject_info)

RNSIDS_PIT=pit_df_subjects.rns_deid_id
#%%
def main():

    for s in range(len(RNSIDS_MGH)):
        print('Running testing for subject ' + RNSIDS_MGH[s] + ' [s]: ' + str(s))
       
        df = pd.read_csv(meta_data_file) 

        test_df = df[df.rns_id==RNSIDS_MGH[s]]
        test_df.reset_index(drop=True, inplace=True)
        
        save_predictions = output_path + RNSIDS_MGH[s]+'/results/'
        if not os.path.exists(save_predictions):
            os.makedirs(save_predictions)
   
        
        test_df['epoch'] = test_df.data.apply(lambda x: x.split("_")[2])
        test_epochs = test_df.epoch.unique()
        
        # prediction saved per PE
        
        for epoch_counter in test_epochs:
    
            # define a test_df_pe
            # the programmming epoch information is in test_df.data
            # "data" column is the file name for the spectrogram
            test_df_pe = test_df[test_df['epoch'] == epoch_counter]
    
            test_data_pe = SeizureDatasetLabelTime(file=test_df_pe,
                                root_dir=SPE_DIR,
                                transform=None,
                                target_transform=None
                                )
    
            # make the prediction for every PIT model
            for ss in range(len(RNSIDS_PIT)):
                
                model = ESPfullnet(hparams['n_cnn_layers'],
                            hparams['n_rnn_layers'],
                            hparams['rnn_dim'],
                            hparams['n_class'],
                            hparams['out_ch'],
                            hparams['dropout'],
                            )
                print('Running model subject ' + RNSIDS_PIT[ss] + ' [ss]: ' + str(ss))
                save_models = save_path + RNSIDS_PIT[ss] +'/models/'
                best_path = save_models + 'model_opt.pth'
                
                # get outputs
                outputs_test = test_model(model, hparams, best_path, test_data_pe)
               
                predict_ = { 
                            "prediction_te": outputs_test,
                            "model": RNSIDS_PIT[ss]
                            }
                np.save(save_predictions+ RNSIDS_MGH[s]+ 'results_pe_'+ str(epoch_counter) + '_model_'+ str(ss) + '.npy', predict_)
    
               
            
        
if __name__=='__main__':
    main()
   