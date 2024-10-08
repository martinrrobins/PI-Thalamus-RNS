import sys
import os

import pandas   as pd
import numpy    as np
import seaborn  as sns

import matplotlib.pyplot        as plt
import matplotlib.font_manager  as fm

from matplotlib.lines     import Line2D
from utilit_espectrograms import get_bool_mask_stim_artifact

sys.path.append(os.path.abspath(os.path.join('..','..','iESPnet_SRC_main','utilities')))
from Generator         import SeizureDatasetLabelTimev2, smoothing_label
from Model             import iESPnet
from TrainEval         import test_model_dsf_iespnet, get_performance_indices

sys.path.append(os.path.abspath(os.path.join('../../..','03 Dynamic-Spatial-Filtering')))
from models            import DynamicSpatialFilter


meta_data_dsf_iespnet = '/media/martin/Disco2/Rns_Data/PITT_PI_EEG/METADATA/allfiles_metadata.csv'
df_meta_dsf_iespnet   = pd.read_csv(meta_data_dsf_iespnet)

spe_dir_dsf_iespnet   = '/media/martin/Disco2/Rns_Data/PITT_PI_EEG/'

patients_dsf_iespnet  = df_meta_dsf_iespnet['rns_id'].unique().tolist()
save_path             = '/media/martin/Disco2/Rns_Data/experimentos/dsf_iespnet_lopo/'

# Variables iESPnet
FREQ_MASK_PARAM       = 10
TIME_MASK_PARAN       = 20
N_CLASSES             = 1
learning_rate_iespnet = 1e-3
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
learning_rate_dsf     = 1e-2

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

best_thr         = 0.2
time             = np.linspace(0, 90, 22500)
time_per_sample  = 90 / 22500

for s in range (len(patients_dsf_iespnet)):

    # define test de df_meta
    test_df  = df_meta_dsf_iespnet[df_meta_dsf_iespnet['rns_id'] == patients_dsf_iespnet[s]].copy()      
    test_df.reset_index(drop = True, inplace=True)

    test_epochs = test_df['data'].apply(lambda x: x.split("_")[2]).unique()

    save_runs        = save_path + patients_dsf_iespnet[s] + '/runs/'
    save_models      = save_path + patients_dsf_iespnet[s] + '/models/'
    save_predictions = save_path + patients_dsf_iespnet[s] + '/results/' 
    save_figs        = save_path + patients_dsf_iespnet[s] + '/figs/'

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
    
    best_path  = save_models + 'model.pth'

    for epoch_counter in test_epochs:
        test_df_pe   = test_df[test_df['data'].apply(lambda x: x.split("_")[2]) == epoch_counter]

        test_data_pe = SeizureDatasetLabelTimev2(
                                                 file             = test_df_pe,
                                                 root_dir         = spe_dir_dsf_iespnet,
                                                 transform        = None,
                                                 target_transform = smoothing_label()  
                                                )
        
        print()
        print('in test: ',patients_dsf_iespnet[s])
        # in test
        outputs_test  = test_model_dsf_iespnet(model1, model2, hparams, best_path, test_data_pe)
        prediction_te = get_performance_indices(outputs_test['y_true'], outputs_test['y_prob'], best_thr)

        total_stim_time = 0
        total_num_stim  = 0
        
        for i in range (len(test_data_pe)):
            ieeg, label         = test_data_pe[i]
            num_stim_samples_ch = 0
            num_stim            = 0
            for j in range (4):
                stim_mask, num_stim_segments= get_bool_mask_stim_artifact(
                                                                          ts                           = ieeg[j,:], 
                                                                          time                         = time, 
                                                                          samples_consecutive_artifact = 12, 
                                                                          samples_skip_rebound         = 500
                                                                         )
    
                num_stim_samples_ch += np.sum(~stim_mask)
                num_stim += num_stim_segments

            total_stim_time_ch = num_stim_samples_ch * time_per_sample
            tiempo_prom_ch     = total_stim_time_ch / ieeg.shape[0]
            num_stim_prom_ch   = num_stim / ieeg.shape[0]

            total_stim_time += tiempo_prom_ch
            total_num_stim  += num_stim_prom_ch
        prom_stim_time = (total_stim_time / len(test_df_pe))
        stim_time      = (total_stim_time / total_num_stim)


        
        predict_ = { 
                    "prediction_te"   : prediction_te,
                    "prom_stim_time"  : prom_stim_time,
                    "stim_time"       : stim_time
                    }

        np.save(save_predictions + str(epoch_counter) + '_' + 'results.npy', predict_)

        del test_data_pe