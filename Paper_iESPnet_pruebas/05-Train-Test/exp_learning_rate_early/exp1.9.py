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
from TrainEval         import train_model_dsf_iespnet_abl_early, test_model_dsf_iespnet_abl, get_performance_indices

sys.path.append(os.path.abspath(os.path.join('../../../..','02 Dynamic-Spatial-Filtering')))
from models            import DynamicSpatialFilter

# direccion donde se encuentran los espectrogramas 
SPE_DIR        = '/home/mrobins/Rns_Data/PITT_PI_EEG/'                                #'/media/martin/Disco2/Rns_Data/PITT_PI_EEG/'
meta_data_file = '/home/mrobins/Rns_Data/PITT_PI_EEG/METADATA/allfiles_metadata.csv'  #'/media/martin/Disco2/Rns_Data/PITT_PI_EEG/METADATA/allfiles_metadata.csv'

df_meta        = pd.read_csv(meta_data_file)

# Variables iESPnet
FREQ_MASK_PARAM       = 10
TIME_MASK_PARAN       = 20
N_CLASSES             = 1
learning_rate_iespnet = 9e-4
batch_size            = 64    #128
epochs                = 100
num_workers           = 4

save_path             = 'dsf_iespnet_lr_early/'
patients              = df_meta['rns_id'].unique().tolist()
patience              = 30

# Variables DSF
denoising             = 'autoreject'   # 'autoreject' 'data_augm' 
model                 = 'stager_net'
dsf_type              = 'dsfd'         # 'dsfd' 'dsfm_st'
mlp_input             = 'log_diag_cov'
dsf_soft_thresh       = False
dsf_n_out_channels    = None
n_channels            = 4
learning_rate_dsf     = 1e-3  


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

# define train y test de df_meta
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

# experimentos que se van a realizar
experiment = 'exp1.9'

def main():
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
            
    save_runs        = save_path + experiment + '/runs/'
    save_models      = save_path + experiment + '/models/'
    save_predictions = save_path + experiment + '/results/'
    save_figs        = save_path + experiment + '/figs/'
    
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
            
    print('Running training for: ' + experiment)
    
    # Dataloaders creados
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
            
    # validation data should be balanced, just be "as it is"
    vali_data  = SeizureDatasetLabelTimev2(
                                           file             = vali_df,
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
    
    outputfile = save_models + 'model_' + experiment
    avg_train_losses, train_accs, avg_valid_losses, valid_accs = train_model_dsf_iespnet_abl_early(
                                                                                                   model1, 
                                                                                                   model2, 
                                                                                                   hparams, 
                                                                                                   epochs, 
                                                                                                   train_data, 
                                                                                                   vali_data, 
                                                                                                   transform_train, 
                                                                                                   sampler, 
                                                                                                   outputfile,
                                                                                                   patience
                                                                                                  )
    
    # visualizacion 
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
    plt.plot(range(1,len(train_accs)+1),train_accs, label='Training AUCpr')
    plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
    plt.plot(range(1,len(valid_accs)+1),valid_accs,label='Validation AUCpr')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Min Val Loss')
    maxposs = valid_accs.index(max(valid_accs))+1 
    plt.axvline(maxposs, linestyle='--', color='k',label='Max Val AUCpr')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_figs + 'loss_plot.png', bbox_inches='tight')
    plt.close()

    best_thr   = 0.2 # f1_; deberiamos agregar la funcion?
    # F1_, best_thr = get_thr_output(rang, outputs_val['l_true'], outputs_val['y_prob'])

    best_model_loss = np.argmin(avg_valid_losses) + 1
    best_path_loss  = outputfile + '_' + str(best_model_loss) + '.pth'

    # get the model with the best AUCpr
    best_model_accs = np.argmax(valid_accs)+ 1
    best_path_accs  = outputfile + '_' + str(best_model_accs) + '.pth'

    print('Performance best model loss')        
    print()
    print('in training')
    # in training
    outputs_train_loss = test_model_dsf_iespnet_abl(model1, model2, hparams, best_path_loss, train_data)
    prediction_tr_loss = get_performance_indices(outputs_train_loss['y_true'], outputs_train_loss['y_prob'], best_thr)

    print()
    print('in validation')    
    # in validation
    outputs_vali_loss  = test_model_dsf_iespnet_abl(model1, model2, hparams, best_path_loss, vali_data)
    prediction_va_loss = get_performance_indices(outputs_vali_loss['y_true'], outputs_vali_loss['y_prob'], best_thr)
    
    print()
    print('in testing')    
    # in testing
    outputs_test_loss  = test_model_dsf_iespnet_abl(model1, model2, hparams, best_path_loss, test_data)
    prediction_te_loss = get_performance_indices(outputs_test_loss['y_true'], outputs_test_loss['y_prob'], best_thr)

    print('Performance best model accs')        
    print()
    print('in training')
    # in training
    outputs_train_accs = test_model_dsf_iespnet_abl(model1, model2, hparams, best_path_accs, train_data)
    prediction_tr_accs = get_performance_indices(outputs_train_accs['y_true'], outputs_train_accs['y_prob'], best_thr)

    print()
    print('in validation')    
    # in validation
    outputs_vali_accs  = test_model_dsf_iespnet_abl(model1, model2, hparams, best_path_accs, vali_data)
    prediction_va_accs = get_performance_indices(outputs_vali_accs['y_true'], outputs_vali_accs['y_prob'], best_thr)
    
    print()
    print('in testing')    
    # in testing
    outputs_test_accs  = test_model_dsf_iespnet_abl(model1, model2, hparams, best_path_accs, test_data)
    prediction_te_accs = get_performance_indices(outputs_test_accs['y_true'], outputs_test_accs['y_prob'], best_thr)
    
    predict_ = { 
                "train_losses"       : avg_train_losses,
                "train_acupr"        : train_accs,
                "valid_losses"       : avg_valid_losses, 
                "valid_acupr"        : valid_accs,
                "prediction_tr_loss" : prediction_tr_loss,
                "prediction_va_loss" : prediction_va_loss, 
                "prediction_te_loss" : prediction_te_loss,
                "prediction_tr_accs" : prediction_tr_accs,
                "prediction_va_accs" : prediction_va_accs, 
                "prediction_te_accs" : prediction_te_accs,
                "hparams"            : hparams,
                "min_loss"           : minposs,
                "best_auc"           : maxposs, 
                "threshold"          : best_thr,
                "outputs_vali_loss"  : outputs_vali_loss,
                "outputs_vali_accs"  : outputs_vali_accs,
               #"F1val"              : F1_, 
                "train_size"         : len(train_data)/len(df_meta) # verificar tamaño de train data
               }
    
    np.save(save_predictions + 'results.npy', predict_)
    
    del train_data, test_data, vali_data, model1, model2
    torch.cuda.empty_cache()
            
if __name__=='__main__':
    main()