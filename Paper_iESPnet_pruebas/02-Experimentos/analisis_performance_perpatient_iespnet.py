import sys
import os
import torch
import random

import torchaudio.transforms as T
import pandas                as pd
import numpy                 as np

from torchvision       import transforms

sys.path.append(os.path.abspath(os.path.join('..','..','iESPnet_SRC_main','utilities')))
from Generator         import SeizureDatasetLabelTime, smoothing_label
from Model             import iESPnet
from TrainEval         import test_model_iespnet, get_performance_indices

sys.path.append(os.path.abspath(os.path.join('../../..','03 Dynamic-Spatial-Filtering')))
from models            import DynamicSpatialFilter

# direccion donde se encuentran los espectrogramas 
SPE_DIR        = '/media/martin/Disco2/Rns_Data/PITT_PI_SPEC/'
meta_data_file = '/media/martin/Disco2/Rns_Data/PITT_PI_SPEC/METADATA/allfiles_metadata.csv'

df_meta        = pd.read_csv(meta_data_file)

# Variables iESPnet
FREQ_MASK_PARAM       = 10
TIME_MASK_PARAN       = 20
N_CLASSES             = 1
learning_rate_iespnet = 1e-3
batch_size            = 64    #128
epochs                = 20
num_workers           = 4

# hiperparametros iESPnet y DSF
hparams = {
           "n_cnn_layers"          : 3,
           "n_rnn_layers"          : 3,
           "rnn_dim"               : [150, 100, 50],
           "n_class"               : N_CLASSES,
           "out_ch"                : [8,8,16],
           "dropout"               : 0.3,
           "learning_rate_iespnet" : learning_rate_iespnet,
           "batch_size"            : batch_size,
           "num_workers"           : num_workers,
           "epochs"                : epochs
          }

model = iESPnet(
                hparams['n_cnn_layers'],
                hparams['n_rnn_layers'],
                hparams['rnn_dim'],
                hparams['n_class'],
                hparams['out_ch'],
                hparams['dropout'],
               )

save_path        = '/media/martin/Disco2/Rns_Data/experimentos/iespnet_global/'
experiment       = 'exp3.2'
save_models      = save_path + experiment + '/models/'
save_predictions = save_path + experiment + '/results/'
outputfile       = save_models + 'model_' + experiment
best_path        = outputfile + '.pth'

test_id  = ['PIT-RNS1090', 'PIT-RNS8973', 'PIT-RNS1438', 'PIT-RNS8326', 'PIT-RNS3016']
vali_id  = ['PIT-RNS1603', 'PIT-RNS1556', 'PIT-RNS1534', 'PIT-RNS6989', 'PIT-RNS2543', 'PIT-RNS7168', 'PIT-RNS6762']


train_df = df_meta.copy()
test_df  = pd.DataFrame()
vali_df  = pd.DataFrame()

for s in range (len(test_id)):
    test_df = pd.concat([test_df, df_meta[df_meta['rns_id'] == test_id[s]]])
    test_df.reset_index(drop=True, inplace=True)
    train_df.drop(train_df[train_df['rns_id'] == test_id[s]].index, inplace = True)

for s in range(len(vali_id)):
    vali_df=pd.concat([vali_df, df_meta[df_meta['rns_id'] == vali_id[s]]])
    vali_df.reset_index(drop=True, inplace=True)
    train_df.drop(train_df[train_df['rns_id'] == vali_id[s]].index, inplace = True)

patients_train = train_df['rns_id'].unique().tolist()
patients_test  = test_df['rns_id'].unique().tolist()
patients_vali  = vali_df['rns_id'].unique().tolist()

def main():
    # set the seed for reproducibility
    torch.manual_seed(0)
    random.seed(0)

    best_thr = 0.2

    for s in range (len(patients_train)):
        # Dataloaders creados
        train_data = SeizureDatasetLabelTime(
                                             file             = train_df[train_df['rns_id'] == patients_train[s]],
                                             root_dir         = SPE_DIR,
                                             transform        = None, 
                                             target_transform = smoothing_label(),
                                            )
    
        print()
        print('in training: ', patients_train[s] )
        # in training
        outputs_train = test_model_iespnet(model, hparams, best_path, train_data)
        prediction_tr = get_performance_indices(outputs_train['y_true'], outputs_train['y_prob'], best_thr)

        predict_ = { 
                    "prediction_tr"   : prediction_tr,
                   }
    
        np.save(save_predictions + patients_train[s] + '_results.npy', predict_)

        del train_data

    for s in range (len(patients_vali)):
        # testing data should be balanced, just be "as it is"
        vali_data  = SeizureDatasetLabelTime(
                                             file             = vali_df[vali_df['rns_id'] == patients_vali[s]],
                                             root_dir         = SPE_DIR,
                                             transform        = None,
                                             target_transform = smoothing_label()  
                                            )
    
        print()
        print('in validation: ',patients_vali[s])
        # in validation
        outputs_vali = test_model_iespnet(model, hparams, best_path, vali_data)
        prediction_va = get_performance_indices(outputs_vali['y_true'], outputs_vali['y_prob'], best_thr)

        predict_ = { 
                    "prediction_va"   : prediction_va,
                   }
    
        np.save(save_predictions + patients_vali[s] + '_results.npy', predict_)

        del vali_data

    for s in range (len(patients_test)):
        # testing data should be balanced, just be "as it is"
        test_data  = SeizureDatasetLabelTime(
                                             file             = test_df[test_df['rns_id'] == patients_test[s]],
                                             root_dir         = SPE_DIR,
                                             transform        = None,
                                             target_transform = smoothing_label()  
                                            )
        
        print()
        print('in testing: ', patients_test[s])
        # in testing
        outputs_test  = test_model_iespnet(model, hparams, best_path, test_data)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)
    
        predict_ = { 
                    "prediction_te"   : prediction_te,
                   }
        
        np.save(save_predictions + patients_test[s] + '_results.npy', predict_)

        del test_data
        
    torch.cuda.empty_cache()
            
if __name__=='__main__':
    main()