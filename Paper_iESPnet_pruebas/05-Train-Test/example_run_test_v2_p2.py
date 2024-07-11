"""
Train and test the model 
Training is based on held-one-patient-out (patient 0)

"""
import sys
import os 
import torch
import pandas as pd
import torchaudio.transforms as T
import torch.optim as optim
import numpy as np

# no se sabe si se necesitan
import gc
import multiprocessing

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



sys.path.append(os.path.abspath(os.path.join('..','..','iESPnet_SRC_main','utilities')))
from Generator import SeizureDatasetLabelTime, scale_spec, permute_spec, smoothing_label
from Model import iESPnet
from TrainEval import train_model_opt, test_model, train_model, get_thr_output, get_performance_indices
from utilit_train_test import make_weights_for_balanced_classes


# set the seed for reproducibility
torch.manual_seed(0)

# direccion donde se encuentran los espectrogramas (path: martin)
SPE_DIR  = '/home/mrobins/Rns_Data/PITT_PI_v2/'


# get metadata file
meta_data_file = '/home/mrobins/Rns_Data/PITT_PI_v2/METADATA_v2/allfiles_metadata.csv'


df_meta = pd.read_csv(meta_data_file)



FREQ_MASK_PARAM = 10
TIME_MASK_PARAN = 20
N_CLASSES       = 1
learning_rate   = 1e-3
batch_size      = 128
epochs          = 20
num_workers     = 4
save_path       = 'SAVEPATH_v2_bis/'
patients        = df_meta['rns_id'].unique().tolist()


hparams = {
        "n_cnn_layers" : 3,
        "n_rnn_layers" : 3,
        "rnn_dim"      : [150, 100, 50],
        "n_class"      : N_CLASSES,
        "out_ch"       : [8,8,16],
        "dropout"      : 0.3,
        "learning_rate": learning_rate,
        "batch_size"   : batch_size,
        "num_workers"  : num_workers,
        "epochs"       : epochs
        }

def main():

    for s in range (15,len(patients)):
        model = iESPnet(hparams['n_cnn_layers'],
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
        test_df  = df_meta[df_meta['rns_id'] == patients[s]]
        test_df.reset_index(drop=True, inplace=True)
        train_df.drop(train_df[train_df['rns_id'] == patients[s]].index, inplace = True)


        # Dataloaders creados
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

        # se debe balancear train_df
        weights = make_weights_for_balanced_classes(train_df, [0,1], n_concat=2)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        outputfile = save_models + 'model'
        avg_train_losses, avg_train_f1= train_model_opt(model, hparams, epochs, train_data, sampler, outputfile)
                
        best_thr = 0.2
        best_path = save_models + 'model_opt.pth'
                
        # in testing
        outputs_test=test_model(model, hparams, best_path, test_data)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)
                
        # in training
        outputs_train=test_model(model, hparams, best_path, train_data_ori)
        prediction_tr = get_performance_indices(outputs_train['y_true'], outputs_train['y_prob'], best_thr)
                
        predict_ = { 
                    "train_losses" : avg_train_losses,
                    "train_acupr"  : avg_train_f1,
                    "prediction_te": prediction_te,
                    "prediction_tr": prediction_tr, 
                    "hparams"      : hparams, 
                    "threshold"    : 0.2, 
                    "train_size"   : len(train_data_ori)/len(df_meta)

                    }
        np.save(save_predictions+ patients[s]+ 'results.npy', predict_)
                

        del train_data, test_data

if __name__=='__main__':
    main()
