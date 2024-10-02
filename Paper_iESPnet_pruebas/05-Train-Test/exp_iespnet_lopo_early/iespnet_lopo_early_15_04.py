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
from Generator         import SeizureDatasetLabelTime, permute_spec, smoothing_label
from Model             import iESPnet
from TrainEval         import train_model_iespnet_abl_early, test_model_iespnet, get_performance_indices

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
save_path       = 'iespnet_lopo_early_15/'
patients        = df_meta['rns_id'].unique().tolist()
patience        = 30

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

np.random.seed(3)

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
        test_patient  = patients[s]
        patients_lopo = [p for p in patients if p != test_patient]
        vali_patient  = np.random.choice(patients_lopo)

        train_df = df_meta.copy()
        train_df.drop(train_df[train_df['rns_id'] == vali_patient].index, inplace=True)
        train_df.drop(train_df[train_df['rns_id'] == patients[s]].index, inplace = True)


        test_df  = df_meta[df_meta['rns_id'] == patients[s]].copy()      
        vali_df  = df_meta[df_meta['rns_id'] == vali_patient].copy()
        
        # resetear los índices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        vali_df.reset_index(drop=True, inplace=True)

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
            
        # validation data should be balanced, just be "as it is"
        vali_data  = SeizureDatasetLabelTime(
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

        avg_train_losses, train_accs, avg_valid_losses, valid_accs = train_model_iespnet_abl_early(
                                                                                                   model, 
                                                                                                   hparams, 
                                                                                                   epochs, 
                                                                                                   train_data, 
                                                                                                   vali_data, 
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

        best_thr   = 0.2

        best_model_loss = np.argmin(avg_valid_losses) + 1
        best_path_loss  = outputfile + '_' + str(best_model_loss) + '.pth'

        # get the model with the best AUCpr
        best_model_accs = np.argmax(valid_accs)+ 1
        best_path_accs  = outputfile + '_' + str(best_model_accs) + '.pth'

        print('Performance best model loss')        
        print()
        print('in training')
        # in training
        outputs_train_loss = test_model_iespnet(model, hparams, best_path_loss, train_data)
        prediction_tr_loss = get_performance_indices(outputs_train_loss['y_true'], outputs_train_loss['y_prob'], best_thr)

        print()
        print('in validation')    
        # in validation
        outputs_vali_loss  = test_model_iespnet(model, hparams, best_path_loss, vali_data)
        prediction_va_loss = get_performance_indices(outputs_vali_loss['y_true'], outputs_vali_loss['y_prob'], best_thr)
    
        print()
        print('in testing')    
        # in testing
        outputs_test_loss  = test_model_iespnet(model, hparams, best_path_loss, test_data)
        prediction_te_loss = get_performance_indices(outputs_test_loss['y_true'], outputs_test_loss['y_prob'], best_thr)
        
        print()
        print('Performance best model accs')        
        print()
        print('in training')
        # in training
        outputs_train_accs = test_model_iespnet(model, hparams, best_path_accs, train_data)
        prediction_tr_accs = get_performance_indices(outputs_train_accs['y_true'], outputs_train_accs['y_prob'], best_thr)

        print()
        print('in validation')    
        # in validation
        outputs_vali_accs  = test_model_iespnet(model, hparams, best_path_accs, vali_data)
        prediction_va_accs = get_performance_indices(outputs_vali_accs['y_true'], outputs_vali_accs['y_prob'], best_thr)
    
        print()
        print('in testing')    
        # in testing
        outputs_test_accs  = test_model_iespnet(model, hparams, best_path_accs, test_data)
        prediction_te_accs = get_performance_indices(outputs_test_accs['y_true'], outputs_test_accs['y_prob'], best_thr)
        print()

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
                    "vali_patient"       : vali_patient,
                   #"F1val"              : F1_, 
                    "train_size"         : len(train_data)/len(df_meta) # verificar tamaño de train data
                   }

        np.save(save_predictions + 'results.npy', predict_)
    
        del train_data, test_data, vali_data
        torch.cuda.empty_cache() 
        
if __name__=='__main__':
    main()
