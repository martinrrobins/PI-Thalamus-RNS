#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and test the model the model
Training is based on held-one-patient-out
@author: vpeterson
"""
import os
import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'utilities')))
from Generator import SeizureDatasetLabelTime, scale_spec, permute_spec, smoothing_label
from Model import iESPnet
from TrainEval import train_model_opt, test_model, train_model, get_thr_output, get_performance_indices
import IO
import torchaudio.transforms as T
import pandas as pd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gc
import multiprocessing
# from torchsummary import summary
# set the seed for reproducibility
torch.manual_seed(0)
#%%
def get_class_weights(train_df, classes):
    class_sample_count = np.zeros(len(classes,), dtype=int)
    for n, cl in enumerate(classes):
        class_sample_count[n] = sum(train_df.label==cl)
    class_weights = class_sample_count / sum(class_sample_count)             
    return class_weights

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

def get_val_subject(df, s_te, min_sz, n_sv=2):
    import random    
    sv = []
    a = np.arange(0, len(RNSIDS))
    a = np.ma.array(a, mask=False)
    for i in range(n_sv):
        if i==0:
            # delete testing subject first 
            a.mask[s_te] = True
            b = a.compressed()
            sv_aux =  random.choice([s for s in b if (sum(df[df.rns_id==RNSIDS[s]].label)>min_sz) and sum(df[df.rns_id==RNSIDS[s]].label) < 1000])
        else:
            
            a.mask[sv_aux] = True
            b = a.compressed()
            sv_aux =  random.choice([s for s in b if sum(df[df.rns_id==RNSIDS[s]].label)>min_sz])
            
        sv.append(sv_aux)
    return sv

def plot_outputs(prediction, k, save_figs):
    labels = prediction['l_true']
    target = prediction['y_true']
    predicted = prediction['y_pred']
    
    idx_tp = np.where(labels==1)[0]
    idx_tn = np.where(labels==0)[0]

    if k > len(idx_tp):
        k = len(idx_tp)

    sample_tp = np.random.choice(idx_tp, k, replace=False)
    sample_tn = np.random.choice(idx_tn, k, replace=False)


    for ii in range(k):
            
        ax1 = plt.subplot(211)
    
        ax1.plot(target[sample_tp[ii]], label = 'true')
        ax1.plot(predicted[sample_tp[ii]], label= 'estimated')
        ax1.set_title('pos class sample ' + str(sample_tp[ii]))
        
        ax2 = plt.subplot(212)
        ax2.set_yticks([0,1])
        ax2.plot(target[sample_tn[ii]], label = 'true')
        ax2.plot(predicted[sample_tn[ii]], label= 'estimated')
        ax2.set_title('neg class sample ' + str(sample_tn[ii]))
        plt.legend()
        plt.savefig(save_figs+'/predic_plot_' + str(ii)+'.png', bbox_inches='tight')
        plt.close()
#%%
DATA_DIR = "X:/RNS_DataBank/PITT/"
# If you want to run this code, I recommend you have a local copy of the spectrograms
# to have the script run faster.
SPE_DIR = '../iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll/PITT/'
# get metadata file
meta_data_file = '../iESPnet/Data/RNS_Databank_Spectrograms/TimeLabelZeropadAll/PITT/METADATA/allfiles_nothalamus_metadata.csv'
df = pd.read_csv(meta_data_file) 

FREQ_MASK_PARAM = 10
TIME_MASK_PARAN = 20
N_CLASSES = 1
learning_rate = 1e-3
batch_size = 128
epochs = 20
num_workers = 4
save_path = 'SAVEPATH'
df_subjects = pd.read_csv('../iESPnet/Data/Metadatafiles/subjects_info_zeropadall_nothalamus.csv')

RNSIDS=df_subjects.rns_deid_id
# save_path_hoo = 'C:/Users/vp820/Documents/ESPnet/outputs/HOO/SVzeropad/'

#%%
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


def main():
    # len(RNSIDS)
    for s in range (len(RNSIDS)):
        
        model = iESPnet(hparams['n_cnn_layers'],
                       hparams['n_rnn_layers'],
                       hparams['rnn_dim'],
                       hparams['n_class'],
                       hparams['out_ch'],
                       hparams['dropout'],
                       )
        
        save_runs = save_path + RNSIDS[s] + '/runs/'
        save_models = save_path + RNSIDS[s] +'/models/'
        save_predictions = save_path + RNSIDS[s]+'/results/'
        save_figs = save_path +RNSIDS[s]+'/figs/'
        
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if not os.path.exists(save_runs):
            os.makedirs(save_runs)
            
        if not os.path.exists(save_models):
            os.makedirs(save_models)
            
        if not os.path.exists(save_predictions):
            os.makedirs(save_predictions)
            
        if not os.path.exists(save_figs):
            os.makedirs(save_figs)
    
    
        print('Running training for subject ' + RNSIDS[s] + ' [s]: ' + str(s))
       
        
        train_df = df.copy()
        # define train, val and test from df
        test_df = df[df.rns_id==RNSIDS[s]]
        test_df.reset_index(drop=True, inplace=True)
    
        train_df.drop(train_df[train_df['rns_id'] == RNSIDS[s]].index, inplace = True)
            
        # DATA LOADERS
        train_data_ori = SeizureDatasetLabelTime(file=train_df,
                                    root_dir=SPE_DIR,
                                    transform=None, 
                                    target_transform=smoothing_label(),
                                    )
       
    
        transform_train1 = transforms.Compose([T.FrequencyMasking(FREQ_MASK_PARAM),
                                           T.TimeMasking(TIME_MASK_PARAN), 
                                           permute_spec()
                                           
                                           
                                          ])
        
        # data augmentation only in train data
        train_data_trf1 = SeizureDatasetLabelTime(file=train_df,
                                                root_dir=SPE_DIR,
                                                transform=transform_train1, 
                                                target_transform=smoothing_label() 
                                                )
        
        train_data = torch.utils.data.ConcatDataset([train_data_ori, train_data_trf1])
        # testing data should be balanced, just be "as it is"
        test_data = SeizureDatasetLabelTime(file=test_df,
                                root_dir=SPE_DIR,
                                transform=None,
                                target_transform=smoothing_label()  
                                )
    
        # weights for classes
        weights = make_weights_for_balanced_classes(train_df, [0,1], n_concat=2)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        
        if len(weights) != len(train_data):
            AssertionError('sampler should be equal to train data shape')
        #%% train
        outputfile = save_models + 'model'
        avg_train_losses, avg_train_f1= train_model_opt(model, hparams, epochs, train_data, sampler, outputfile)
                                                            
        #%% eval        
        best_thr = 0.2
        best_path = save_models + 'model_opt.pth'
                
        # in testing
        outputs_test=test_model(model, hparams, best_path, test_data)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)
                
        # in training
        outputs_train=test_model(model, hparams, best_path, train_data_ori)
        prediction_tr = get_performance_indices(outputs_train['y_true'], outputs_train['y_prob'], best_thr)
                
        predict_ = { 
                    "train_losses": avg_train_losses,
                    "train_acupr": avg_train_f1,
                    "prediction_te": prediction_te,
                    "prediction_tr": prediction_tr, 
                    "hparams": hparams, 
                    "threshold": 0.2, 
                    "train_size": len(train_data_ori)/len(df)

                    }
        np.save(save_predictions+ RNSIDS[s]+ 'results.npy', predict_)
                
        
        del train_data, test_data

if __name__=='__main__':
    main()
