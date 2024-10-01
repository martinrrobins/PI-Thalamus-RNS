import sys
import os
import torch
import random
import gc

import matplotlib.pyplot        as plt
import torchaudio.transforms    as T
import torch.optim              as optim
import pandas                   as pd
import numpy                    as np

from torchvision       import transforms

sys.path.append(os.path.abspath(os.path.join('..')))
from utilit_train_test import make_weights_for_balanced_classes

sys.path.append(os.path.abspath(os.path.join('..','..','..','iESPnet_SRC_main','utilities')))
from Generator         import SeizureDatasetLabelTimev2, permute_spec, smoothing_label
from Model             import iESPnet
from TrainEval         import train_model_dsf_iespnet, test_model_dsf_iespnet, get_performance_indices

sys.path.append(os.path.abspath(os.path.join('../../../..','02 Dynamic-Spatial-Filtering')))
from models            import DynamicSpatialFilter

# direccion donde se encuentran los datos 
SPE_DIR        = '/home/mrobins/Rns_Data/PITT_PI_EEG/'                                #'/media/martin/Disco2/Rns_Data/PITT_PI_EEG/'
meta_data_file = '/home/mrobins/Rns_Data/PITT_PI_EEG/METADATA/allfiles_metadata.csv'  #'/media/martin/Disco2/Rns_Data/PITT_PI_EEG/METADATA/allfiles_metadata.csv'

df_meta        = pd.read_csv(meta_data_file)

# Variables iESPnet
FREQ_MASK_PARAM       = 10
TIME_MASK_PARAN       = 20
N_CLASSES             = 1
learning_rate_iespnet = 7e-4
batch_size            = 64    #128
epochs                = 20
num_workers           = 4

# Variables DSF
denoising             = 'autoreject'   # 'autoreject' 'data_augm' 
model                 = 'stager_net'
dsf_type              = 'dsfd'         # 'dsfd' 'dsfm_st'
mlp_input             = 'log_diag_cov'
dsf_soft_thresh       = False
dsf_n_out_channels    = None
n_channels            = 4
learning_rate_dsf     = 1e-3  

save_path             = 'dsf_iespnet_lopo_lr7/'
patients              = df_meta['rns_id'].unique().tolist()

# hiperparametros iESPnet y DSF
hparams = {
           "n_cnn_layers"          : 3,
           "n_rnn_layers"          : 3,
           "rnn_dim"               : [150, 100, 50],
           "n_class"               : N_CLASSES,
           "out_ch"                : [8,8,16],
           "dropout"               : 0.3,
           "learning_rate_iespnet" : learning_rate_iespnet,
           "learning_rate_dsf"     : learning_rate_dsf,
           "batch_size"            : batch_size,
           "num_workers"           : num_workers,
           "epochs"                : epochs
          }

def main():
    for s in range (7,14):
    
        # set the seed for reproducibility
        torch.manual_seed(0)
        random.seed(0)

        model1 = DynamicSpatialFilter(
                                      n_channels, 
                                      mlp_input            = mlp_input, 
                                      n_out_channels       = dsf_n_out_channels, 
                                      apply_soft_thresh    = dsf_soft_thresh
                                     )
            
        model2 = iESPnet(
                         hparams['n_cnn_layers'],
                         hparams['n_rnn_layers'],
                         hparams['rnn_dim'],
                         hparams['n_class'],
                         hparams['out_ch'],
                         hparams['dropout'],
                        )
            
        save_runs        = save_path + patients[s] + '/runs/'
        save_models      = save_path + patients[s] + '/models/'
        save_predictions = save_path + patients[s] + '/results/'
        save_figs        = save_path + patients[s] + '/figs/'

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

        print('Running training for subject ' + patients[s] + ' [s]: ' + str(s))

        # define train y test de df_meta
        train_df = df_meta.copy()
        train_df.drop(train_df[train_df['rns_id'] == patients[s]].index, inplace = True)
        train_df.reset_index(drop = True, inplace=True)

        test_df  = df_meta[df_meta['rns_id'] == patients[s]].copy()      
        test_df.reset_index(drop = True, inplace=True)

        # dataloaders creados
        train_data = SeizureDatasetLabelTimev2(
                                               file             = train_df,
                                               root_dir         = SPE_DIR,
                                               transform        = None, 
                                               target_transform = smoothing_label(),
                                              )
            
        # testing data should be balanced, just be "as it is"
        test_data  = SeizureDatasetLabelTimev2(
                                               file             = test_df,
                                               root_dir         = SPE_DIR,
                                               transform        = None,
                                               target_transform = smoothing_label()  
                                              )
            
        # data augmentation 
        transform_train = transforms.Compose([
                                              T.FrequencyMasking(FREQ_MASK_PARAM),
                                              T.TimeMasking(TIME_MASK_PARAN), 
                                              permute_spec()                                                                     
                                            ])
        
        weights = make_weights_for_balanced_classes(train_df, [0,1], n_concat=1)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        outputfile = save_models + 'model'

        avg_train_losses, avg_valid_losses = train_model_dsf_iespnet(
                                                                     model1, 
                                                                     model2, 
                                                                     hparams, 
                                                                     epochs, 
                                                                     train_data, 
                                                                     transform_train, 
                                                                     sampler, 
                                                                     outputfile,
                                                                    )
        
        best_thr = 0.2
        best_path = save_models + 'model.pth'

        print('Performance model')        
        print()
        print('in training')
        # in training
        outputs_train = test_model_dsf_iespnet(model1, model2, hparams, best_path, train_data)
        prediction_tr = get_performance_indices(outputs_train['y_true'], outputs_train['y_prob'], best_thr)
   
        print()
        print('in testing')    
        # in testing
        outputs_test  = test_model_dsf_iespnet(model1, model2, hparams, best_path, test_data)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)
                
        predict_ = { 
                    "train_losses"  : avg_train_losses,
                    "valid_losses"  : avg_valid_losses, 
                    "prediction_tr" : prediction_tr,
                    "prediction_te" : prediction_te,
                    "hparams"       : hparams,
                    "threshold"     : best_thr,
                   #"F1val"         : F1_, 
                    "train_size"    : len(train_data)/len(df_meta) # verificar tamaño de train data
                   }
    
        np.save(save_predictions + 'results.npy', predict_)
    
        del train_data, test_data, model1, model2
        torch.cuda.empty_cache() 
        
if __name__=='__main__':
    main()