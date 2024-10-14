import sys
import os
import torch
import random
import gc

import torchaudio.transforms    as T
import torch.optim              as optim
import pandas                   as pd
import numpy                    as np

from torchvision       import transforms

sys.path.append(os.path.abspath(os.path.join('..')))
from utilit_train_test import make_weights_for_balanced_classes

sys.path.append(os.path.abspath(os.path.join('..','..','..','iESPnet_SRC_main','utilities')))
from Generator         import SeizureDatasetLabelTime, permute_spec_iespnet, smoothing_label
from Model             import iESPnet
from TrainEval         import train_model_iespnet_lopo, test_model_iespnet, get_performance_indices

# direccion donde se encuentran los espectrogramas (path: martin)
SPE_DIR        = '/home/mrobins/Rns_Data/PITT_PI_SPEC/'                                #'/media/martin/Disco2/Rns_Data/PITT_PI_EEG/'
meta_data_file = '/home/mrobins/Rns_Data/PITT_PI_SPEC/METADATA/allfiles_metadata.csv'  #'/media/martin/Disco2/Rns_Data/PITT_PI_EEG/METADATA/allfiles_metadata.csv'

df_meta = pd.read_csv(meta_data_file)

# Variables iESPnet
FREQ_MASK_PARAM = 10
TIME_MASK_PARAN = 20
N_CLASSES       = 1
learning_rate   = 1e-3
batch_size      = 128
epochs          = 20
num_workers     = 4
save_path       = 'iespnet_lopo/'
patients        = df_meta['rns_id'].unique().tolist()

# hiperparametros iESPnet
hparams = {
        "n_cnn_layers"  : 3,
        "n_rnn_layers"  : 3,
        "rnn_dim"       : [150, 100, 50],
        "n_class"       : N_CLASSES,
        "out_ch"        : [8,8,16],
        "dropout"       : 0.3,
        "learning_rate" : learning_rate,
        "batch_size"    : batch_size,
        "num_workers"   : num_workers,
        "epochs"        : epochs
        }


def main():
    for s in range (21,len(patients)):
        # set the seed for reproducibility
        torch.manual_seed(0)
        random.seed(0)
    
        model = iESPnet(
                        hparams['n_cnn_layers'],
                        hparams['n_rnn_layers'],
                        hparams['rnn_dim'],
                        hparams['n_class'],
                        hparams['out_ch'],
                        hparams['dropout'],
                       )

        save_runs        = save_path + patients[s] +  '/runs/'
        save_models      = save_path + patients[s] +  '/models/'
        save_predictions = save_path + patients[s] +  '/results/'
        save_figs        = save_path + patients[s] +  '/figs/'

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

        # Dataloaders creados
        train_data_orig = SeizureDatasetLabelTime(
                                                  file             = train_df,
                                                  root_dir         = SPE_DIR,
                                                  transform        = None, 
                                                  target_transform = smoothing_label(),
                                                 )
        
        # testing data should be balanced, just be "as it is"
        test_data  = SeizureDatasetLabelTime(
                                             file             = test_df,
                                             root_dir         = SPE_DIR,
                                             transform        = None,
                                             target_transform = smoothing_label()  
                                            )
        
        # data augmentation 
        transform_train = transforms.Compose([
                                              T.FrequencyMasking(FREQ_MASK_PARAM),
                                              T.TimeMasking(TIME_MASK_PARAN), 
                                              permute_spec_iespnet()                                                                     
                                            ])
        
        # data augmentation only in train data
        train_data_tran = SeizureDatasetLabelTime(
                                                  file             = train_df,
                                                  root_dir         = SPE_DIR,
                                                  transform        = transform_train, 
                                                  target_transform = smoothing_label() 
                                                 )
        
        train_data = torch.utils.data.ConcatDataset([train_data_orig, train_data_tran])

        # se debe balancear train_df
        weights = make_weights_for_balanced_classes(train_df, [0,1], n_concat=2)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        outputfile = save_models + 'model'
        
        avg_train_losses, train_accs = train_model_iespnet_lopo(
                                                                model,                                                                                     
                                                                hparams, 
                                                                epochs, 
                                                                train_data,
                                                                sampler, 
                                                                outputfile                                                                                         
                                                               )
                
        best_thr  = 0.2
        best_path = outputfile + '.pth'

        print()
        print('in training')
        # in training
        outputs_train = test_model_iespnet(model, hparams, best_path, train_data)
        prediction_tr = get_performance_indices(outputs_train['y_true'], outputs_train['y_prob'], best_thr)

        print()
        print('in testing')
        # in testing
        outputs_test  = test_model_iespnet(model, hparams, best_path, test_data)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)
            
        predict_ = { 
                    "train_losses" : avg_train_losses,
                    "train_acupr"  : train_accs,
                    "prediction_te": prediction_te,
                    "prediction_tr": prediction_tr, 
                    "hparams"      : hparams, 
                    "threshold"    : 0.2, 
                    "train_size"   : len(train_data)/len(df_meta) # verificar tamaño de train data
                }
        
        np.save(save_predictions + 'results.npy', predict_)
                
        del train_data, test_data, model
        torch.cuda.empty_cache()

if __name__=='__main__':
    main()