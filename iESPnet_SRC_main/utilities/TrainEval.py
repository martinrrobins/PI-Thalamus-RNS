#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for training, evaluating and testnig the model.

@author: vpeterson
"""

from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, mean_absolute_error, average_precision_score
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'utilities')))
import torch.optim as optim
from torch import nn
# for training the model with early stopping
from pytorchtools import EarlyStopping, EarlyStopping_iespnet
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from IO import get_spectrogram_2

#%%
def train_model(model, hparams, epochs, train_data, val_data, sampler, save_path, figure_path, patience):
    # train model with early stopping
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_train_accs = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    avg_valid_accs = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0, path=save_path)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))
    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark = True


    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size=hparams["batch_size"],
                              sampler=sampler,
                              **kwargs)
    val_loader = DataLoader(val_data, batch_size=hparams["batch_size"], shuffle=False,
                            **kwargs)
    #move model to device
    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'], weight_decay=1e-4)

    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')    
    criterion =nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training(model, device, train_loader, criterion, optimizer, scheduler, epoch, figure_path)
        valid_losses, valid_aucpr = validate(model, device, val_loader, criterion, epoch, figure_path)
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        avg_train_accs.append(train_aucpr)
        avg_valid_accs.append(valid_aucpr)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(epoch, valid_loss, model, optimizer)          
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs


def train_model_opt(model, hparams, epochs, train_data, sampler, save_path):
    # train model until the indicated number of epochs
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_train_accs   = []
  
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))
    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark = True


    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size=hparams["batch_size"],
                              sampler=sampler,
                              **kwargs)
    
    #move model to device
    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'], weight_decay=1e-4)

   
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')    
    criterion =nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training(model, device, train_loader, criterion, optimizer, scheduler, epoch)
        
        train_loss = np.average(train_losses)

        avg_train_losses.append(train_loss)
        
        avg_train_accs.append(train_aucpr)
    print('savinf the model')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path + '_opt.pth') 
    return avg_train_losses, avg_train_accs

def train_model_dsf_iespnet(model1, model2, hparams, epochs, train_data, transform_train, sampler, save_path):
    # train model until the indicated number of epochs
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_train_accs   = []
  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size=hparams["batch_size"], sampler = sampler, **kwargs)
    
    #move model1 to device
    model1.to(device)

    #move model2 to device
    model2.to(device)

    print('Num Model Parameters', sum([param1.nelement() for param1 in model1.parameters()]))
    print('Num Model Parameters', sum([param2.nelement() for param2 in model2.parameters()]))

    # para dsf: weight_decay       = 0.001
    # para dsf: lr                 = 0.0004

    optimizer1 = optim.AdamW(model1.parameters(), hparams['learning_rate_dsf'], weight_decay=1e-4)
    optimizer2 = optim.AdamW(model2.parameters(), hparams['learning_rate_iespnet'], weight_decay=1e-4)

    scheduler1 = optim.lr_scheduler.OneCycleLR(
                                               optimizer1, 
                                               max_lr          = hparams['learning_rate_dsf'],  
                                               steps_per_epoch = int(len(train_loader)),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
    scheduler2 = optim.lr_scheduler.OneCycleLR(
                                               optimizer2, 
                                               max_lr          = hparams['learning_rate_iespnet'], 
                                               steps_per_epoch = int(len(train_loader)*2),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_dsf_iespnet(
                                                         model1, 
                                                         model2, 
                                                         device, 
                                                         train_loader, 
                                                         transform_train, 
                                                         criterion, 
                                                         optimizer1, 
                                                         optimizer2, 
                                                         scheduler1, 
                                                         scheduler2, 
                                                         epoch
                                                        )
        
        train_loss = np.average(train_losses)

        avg_train_losses.append(train_loss)
        avg_train_accs.append(train_aucpr)

    print('saving the model')

    torch.save({
                'epoch': epoch,
                'model_state_dict1': model1.state_dict(),
                'model_state_dict2': model2.state_dict(),
                'optimizer_state_dict1': optimizer1.state_dict(),
                'optimizer_state_dict2': optimizer2.state_dict(),
               }, save_path + '.pth')
    
    return avg_train_losses, avg_train_accs

def train_model_dsf_iespnet_exp(model1, model2, hparams, epochs, train_data, vali_data, transform_train, sampler, save_path, experiment_1, experiment_2):
    # train model until the indicated number of epochs
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    valid_accs       = []
  
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
    valid_loader = DataLoader(vali_data, batch_size = hparams["batch_size"], shuffle = False, **kwargs)
    
    #move model1 to device
    model1.to(device)

    #move model2 to device
    model2.to(device)

    print('Num Model Parameters', sum([param1.nelement() for param1 in model1.parameters()]))
    print('Num Model Parameters', sum([param2.nelement() for param2 in model2.parameters()]))

    # para dsf: weight_decay       = 0.001
    # para dsf: lr                 = 0.0004

    optimizer1 = optim.AdamW(model1.parameters(), hparams['learning_rate'], weight_decay=1e-4)
    optimizer2 = optim.AdamW(model2.parameters(), hparams['learning_rate'], weight_decay=1e-4)

    scheduler1 = optim.lr_scheduler.OneCycleLR(
                                               optimizer1, 
                                               max_lr          = hparams['learning_rate'], 
                                               steps_per_epoch = int(len(train_loader)),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
    scheduler2 = optim.lr_scheduler.OneCycleLR(
                                               optimizer2, 
                                               max_lr          = hparams['learning_rate'], 
                                               steps_per_epoch = int(len(train_loader)*2),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_DSF_iESPnet(
                                                         model1, 
                                                         model2, 
                                                         device, 
                                                         train_loader, 
                                                         transform_train, 
                                                         criterion, 
                                                         optimizer1, 
                                                         optimizer2, 
                                                         scheduler1, 
                                                         scheduler2, 
                                                         epoch,
                                                         experiment_1,
                                                         experiment_2
                                                        )
        
        valid_losses, valid_aucpr = validate_v2(
                                                model1,
                                                model2,
                                                device, 
                                                valid_loader, 
                                                criterion, 
                                                epoch,
                                                experiment_1,
                                                experiment_2
                                               )
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accs.append(train_aucpr)
        valid_accs.append(valid_aucpr)

    print('saving the model')

    torch.save({
                'epoch': epoch,
                'model_state_dict1'    : model1.state_dict(),
                'model_state_dict2'    : model2.state_dict(),
                'optimizer_state_dict1': optimizer1.state_dict(),
                'optimizer_state_dict2': optimizer2.state_dict(),
               }, save_path + '.pth')
    
    del optimizer1, optimizer2, scheduler1, scheduler2
    torch.cuda.empty_cache()
     
    return avg_train_losses, train_accs, avg_valid_losses, valid_accs

def train_model_dsf_iespnet_abl(model1, model2, hparams, epochs, train_data, vali_data, transform_train, sampler, save_path):
    # train model until the indicated number of epochs
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    valid_accs       = []
  
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
    valid_loader = DataLoader(vali_data, batch_size = hparams["batch_size"], shuffle = False, **kwargs)
    
    #move model1 to device
    model1.to(device)

    #move model2 to device
    model2.to(device)

    print('Num Model Parameters', sum([param1.nelement() for param1 in model1.parameters()]))
    print('Num Model Parameters', sum([param2.nelement() for param2 in model2.parameters()]))

    # para dsf: weight_decay       = 0.001
    # para dsf: lr                 = 0.0004

    optimizer1 = optim.AdamW(model1.parameters(), hparams['learning_rate_dsf'], weight_decay = 1e-4)
    optimizer2 = optim.AdamW(model2.parameters(), hparams['learning_rate_iespnet'], weight_decay = 1e-4)

    scheduler1 = optim.lr_scheduler.OneCycleLR(
                                               optimizer1, 
                                               max_lr          = hparams['learning_rate_dsf'], 
                                               steps_per_epoch = int(len(train_loader)),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
    scheduler2 = optim.lr_scheduler.OneCycleLR(
                                               optimizer2, 
                                               max_lr          = hparams['learning_rate_iespnet'], 
                                               steps_per_epoch = int(len(train_loader)*2),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_dsf_iespnet_abl(
                                                             model1, 
                                                             model2, 
                                                             device, 
                                                             train_loader, 
                                                             transform_train, 
                                                             criterion, 
                                                             optimizer1, 
                                                             optimizer2, 
                                                             scheduler1, 
                                                             scheduler2, 
                                                             epoch,
                                                            )
        
        valid_losses, valid_aucpr = validate_abl(
                                                 model1,
                                                 model2,
                                                 device, 
                                                 valid_loader, 
                                                 criterion, 
                                                 epoch
                                                )
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accs.append(train_aucpr)
        valid_accs.append(valid_aucpr)

    print('saving the model')

    torch.save({
                'epoch': epoch,
                'model_state_dict1'    : model1.state_dict(),
                'model_state_dict2'    : model2.state_dict(),
                'optimizer_state_dict1': optimizer1.state_dict(),
                'optimizer_state_dict2': optimizer2.state_dict(),
               }, save_path + '.pth')
    
    del optimizer1, optimizer2, scheduler1, scheduler2
    torch.cuda.empty_cache()
     
    return avg_train_losses, train_accs, avg_valid_losses, valid_accs

def train_model_dsf_iespnet_abl_no_sche(model1, model2, hparams, epochs, train_data, vali_data, transform_train, sampler, save_path):
    # train model until the indicated number of epochs
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    valid_accs       = []
  
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
    valid_loader = DataLoader(vali_data, batch_size = hparams["batch_size"], shuffle = False, **kwargs)
    
    #move model1 to device
    model1.to(device)

    #move model2 to device
    model2.to(device)

    print('Num Model Parameters', sum([param1.nelement() for param1 in model1.parameters()]))
    print('Num Model Parameters', sum([param2.nelement() for param2 in model2.parameters()]))

    # para dsf: weight_decay       = 0.001
    # para dsf: lr                 = 0.0004

    optimizer1 = optim.AdamW(model1.parameters(), hparams['learning_rate_dsf'], weight_decay=1e-4)
    optimizer2 = optim.AdamW(model2.parameters(), hparams['learning_rate_iespnet'], weight_decay=1e-4)

    scheduler2 = optim.lr_scheduler.OneCycleLR(
                                               optimizer2, 
                                               max_lr          = hparams['learning_rate_iespnet'], 
                                               steps_per_epoch = int(len(train_loader)*2),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_dsf_iespnet_abl_no_sche(
                                                                     model1, 
                                                                     model2, 
                                                                     device, 
                                                                     train_loader, 
                                                                     transform_train, 
                                                                     criterion, 
                                                                     optimizer1, 
                                                                     optimizer2,                                                             
                                                                     scheduler2, 
                                                                     epoch,
                                                                    )
        
        valid_losses, valid_aucpr = validate_abl(
                                                 model1,
                                                 model2,
                                                 device, 
                                                 valid_loader, 
                                                 criterion, 
                                                 epoch
                                                )
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accs.append(train_aucpr)
        valid_accs.append(valid_aucpr)

    print('saving the model')

    torch.save({
                'epoch': epoch,
                'model_state_dict1'    : model1.state_dict(),
                'model_state_dict2'    : model2.state_dict(),
                'optimizer_state_dict1': optimizer1.state_dict(),
                'optimizer_state_dict2': optimizer2.state_dict(),
               }, save_path + '.pth')
    
    del optimizer1, optimizer2, scheduler1, scheduler2
    torch.cuda.empty_cache()
     
    return avg_train_losses, train_accs, avg_valid_losses, valid_accs

def train_model_dsf_iespnet_abl_early(model1, model2, hparams, epochs, train_data, vali_data, transform_train, sampler, save_path, patience):
    
    # train model until the indicated number of epochs -- dsf + iespnet 
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    valid_accs       = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience = patience, min_epochs=15, verbose = True, delta=0, path = save_path)
  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
    valid_loader = DataLoader(vali_data, batch_size = hparams["batch_size"], shuffle = False, **kwargs)
    
    #move model1 to device
    model1.to(device)

    #move model2 to device
    model2.to(device)

    print('Num Model Parameters', sum([param1.nelement() for param1 in model1.parameters()]))
    print('Num Model Parameters', sum([param2.nelement() for param2 in model2.parameters()]))

    # para dsf: lr = learning_rate_dsf 
    optimizer1 = optim.AdamW(model1.parameters(), hparams['learning_rate_dsf'], weight_decay=1e-4)
    optimizer2 = optim.AdamW(model2.parameters(), hparams['learning_rate_iespnet'], weight_decay=1e-4)

    scheduler1 = optim.lr_scheduler.OneCycleLR(
                                               optimizer1, 
                                               max_lr          = hparams['learning_rate_dsf'], 
                                               steps_per_epoch = int(len(train_loader)),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
                                              
    scheduler2 = optim.lr_scheduler.OneCycleLR(
                                               optimizer2, 
                                               max_lr          = hparams['learning_rate_iespnet'], 
                                               steps_per_epoch = int(len(train_loader)*2),
                                               epochs          = hparams['epochs'],
                                               anneal_strategy = 'linear'
                                              )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_dsf_iespnet_abl_early(
                                                                   model1, 
                                                                   model2, 
                                                                   device, 
                                                                   train_loader, 
                                                                   transform_train, 
                                                                   criterion, 
                                                                   optimizer1, 
                                                                   optimizer2, 
                                                                   scheduler1, 
                                                                   scheduler2, 
                                                                   epoch                                                                   
                                                                  )
        
        valid_losses, valid_aucpr = validate_abl(
                                                 model1,
                                                 model2,
                                                 device, 
                                                 valid_loader, 
                                                 criterion, 
                                                 epoch
                                                )
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accs.append(train_aucpr)
        valid_accs.append(valid_aucpr)

        # early_stopping needs the validation loss to check if it has decresed, and if it has, it will make a checkpoint of the current model
        early_stopping(epoch, valid_loss, model1, model2, optimizer1, optimizer2)          
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
   
    return avg_train_losses, train_accs, avg_valid_losses, valid_accs

def train_model_iespnet(model, hparams, epochs, train_data, vali_data, sampler, save_path):
    # train model until the indicated number of epochs -- iespnet
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    valid_accs       = []
  
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
    valid_loader = DataLoader(vali_data, batch_size = hparams["batch_size"], shuffle = False, **kwargs)
    
    #move model to device
    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
                                              optimizer, 
                                              max_lr          = hparams['learning_rate'], 
                                              steps_per_epoch = int(len(train_loader)),
                                              epochs          = hparams['epochs'],
                                              anneal_strategy = 'linear'
                                             )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_iespnet(
                                                     model,                                                          
                                                     device, 
                                                     train_loader,                                                      
                                                     criterion, 
                                                     optimizer,  
                                                     scheduler, 
                                                     epoch                                                     
                                                    )
        
        valid_losses, valid_aucpr = validate_iespnet(
                                                     model,                                                
                                                     device, 
                                                     valid_loader, 
                                                     criterion, 
                                                     epoch                                                                                                   
                                                    )
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accs.append(train_aucpr)
        valid_accs.append(valid_aucpr)

    print('saving the model')

    torch.save({
                'epoch': epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, save_path + '.pth')

    del optimizer, scheduler
    torch.cuda.empty_cache()
    
    return avg_train_losses, train_accs, avg_valid_losses, valid_accs

def train_model_iespnet_abl_early(model, hparams, epochs, train_data, vali_data, sampler, save_path, patience):
    # train model until the indicated number of epochs -- iespnet
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    valid_accs       = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping_iespnet(patience = patience, min_epochs=15, verbose = True, delta = 0, path = save_path)
      
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
    valid_loader = DataLoader(vali_data, batch_size = hparams["batch_size"], shuffle = False, **kwargs)
    
    #move model to device
    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
                                              optimizer, 
                                              max_lr          = hparams['learning_rate'], 
                                              steps_per_epoch = int(len(train_loader)),
                                              epochs          = hparams['epochs'],
                                              anneal_strategy = 'linear'
                                             )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_iespnet(
                                                     model,                                                          
                                                     device, 
                                                     train_loader,                                                      
                                                     criterion, 
                                                     optimizer,  
                                                     scheduler, 
                                                     epoch                                                     
                                                    )
        
        valid_losses, valid_aucpr = validate_iespnet(
                                                     model,                                                
                                                     device, 
                                                     valid_loader, 
                                                     criterion, 
                                                     epoch                                                                                                   
                                                    )
        
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accs.append(train_aucpr)
        valid_accs.append(valid_aucpr)

        # early_stopping needs the validation loss to check if it has decresed, and if it has, it will make a checkpoint of the current model
        early_stopping(epoch, valid_loss, model, optimizer)

        if early_stopping.early_stop:
            print("Early stopping")
            break   

    return avg_train_losses, train_accs, avg_valid_losses, valid_accs

def train_model_iespnet_lopo(model, hparams, epochs, train_data, sampler, save_path):
    # train model until the indicated number of epochs -- iespnet
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    train_accs       = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion to speed up training
    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_data, batch_size = hparams["batch_size"], sampler = sampler, **kwargs)
        
    #move model to device
    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
                                              optimizer, 
                                              max_lr          = hparams['learning_rate'], 
                                              steps_per_epoch = int(len(train_loader)),
                                              epochs          = hparams['epochs'],
                                              anneal_strategy = 'linear'
                                             )
          
    criterion = nn.BCEWithLogitsLoss().to(device)
 
    for epoch in range(1, epochs + 1):
        train_losses, train_aucpr = training_iespnet(
                                                     model,                                                          
                                                     device, 
                                                     train_loader,                                                      
                                                     criterion, 
                                                     optimizer,  
                                                     scheduler, 
                                                     epoch                                                     
                                                    )
        
        
        train_loss = np.average(train_losses)

        avg_train_losses.append(train_loss)
        
        train_accs.append(train_aucpr)

    print('saving the model')

    torch.save({
                'epoch': epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, save_path + '.pth')

    del optimizer, scheduler
    torch.cuda.empty_cache()
    
    return avg_train_losses, train_accs

def training(model, device, train_loader, criterion, optimizer, scheduler, epoch, figure_path=[]):
    data_len = len(train_loader.dataset)
    train_loss = 0.0
    
    # train with early stopping
    # to track the training loss as the model trains
    train_losses = []
    # precision = Precision(average=False, device=device)
    # recall = Recall(average=False, device=device)
    cont = 0
    model.train()
    for batch_idx, _data in enumerate(train_loader):
        cont+=1
        # print(batch_idx)
        spectrograms, labels = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)
        # Perform forward pass
        outputs = model(spectrograms)  # (batch, n_class) 
        
        m = nn.Sigmoid()
        probs = m(outputs)
        
        y_true  = torch.max(labels, dim =1)[0]
        y_pred = torch.max(probs, dim=1)[0]
        
        if cont==1:
            Y_true = y_true
            Y_pred= y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)


        # Compute loss
        loss = criterion(outputs, labels)
        # Perform backward pass
        loss.backward()
        train_loss += loss.item()
        # Perform optimization
        optimizer.step()
        scheduler.step()
        
        # record training loss
        train_losses.append(loss.item())
    

        # if batch_idx % 100 == 99:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(spectrograms), data_len,
        #         100. * batch_idx / len(train_loader), train_loss/100))
        #     train_loss = 0.0 
        del _data
        torch.cuda.empty_cache()
    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()
    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr= average_precision_score(y_val_true,val_pred)
    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))
  
    return train_losses, train_aucpr

def training_dsf_iespnet(model1, model2 ,device, train_loader, transform_train, criterion, optimizer1, optimizer2, scheduler1, scheduler2, epoch):  
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    top_db           = 60.0


    train_loss = 0.0

    # train with early stopping to track the training loss as the model trains
    train_losses = []
    # precision = Precision(average=False, device=device)
    # recall    = Recall(average=False, device=device)

    cont = 0

    model1.train()
    model2.train()

    #train_loader es un dataloader que devuelve el eeg y el label continuo suavizado, es decir con el smoothing_label
    for batch_idx, _data in enumerate(train_loader):

        cont+=1
        eeg, labels = _data 
        eeg, labels = eeg.to(device), labels.to(device)

        # Zero the gradients
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        # Perform forward pass to DSF
        outputs1 = model1(eeg)  # (batch, n_class)
        outputs1 = outputs1.squeeze(1)
        outputs1 = (outputs1 - outputs1.mean()) / outputs1.std()
        outputs1 = outputs1.to('cpu')

        # create spectrogram from outputs1
        spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
        spectrograms = torch.from_numpy(spectrograms)

        spectrograms_transformed = transform_train(spectrograms) 

        # contact tensors
        spectrograms2train = torch.cat((spectrograms, spectrograms_transformed), axis=0)
        spectrograms2train = spectrograms2train.to(device)

        # las labels tb se duplican
        labels2train       = torch.cat((labels, labels), axis=0)

        # Perform forward pass to iESPnet
        outputs2 = model2(spectrograms2train)

        m     = nn.Sigmoid()
        probs = m(outputs2)
        
        y_true  = torch.max(labels2train, dim = 1)[0]
        y_pred  = torch.max(probs, dim = 1)[0]
        
        if cont == 1:
            Y_true = y_true
            Y_pred = y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)


        # Compute loss
        loss = criterion(outputs2, labels2train)

        # Perform backward pass
        loss.backward()
        train_loss += loss.item()

        # Perform optimization
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        
        # record training loss
        train_losses.append(loss.item())

        del _data
        torch.cuda.empty_cache()

    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()

    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr = average_precision_score(y_val_true,val_pred)
        
    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))

    return train_losses, train_aucpr

def training_DSF_iESPnet(model1, model2 ,device, train_loader, transform_train, criterion, optimizer1, optimizer2, scheduler1, scheduler2, epoch, experiment_1, experiment_2):  
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    if   experiment_2 == '.1':  
        top_db       = 40.0
    elif experiment_2 == '.2':
        top_db       = 60.0
    elif experiment_2 == '.3':
        top_db       = 80.0

    train_loss = 0.0

    # train with early stopping to track the training loss as the model trains
    train_losses = []
    # precision = Precision(average=False, device=device)
    # recall    = Recall(average=False, device=device)

    cont = 0

    model1.train()
    model2.train()

    #train_loader es un dataloader que devuelve el eeg y el label continuo suavizado, es decir con el smoothing_label
    for batch_idx, _data in enumerate(train_loader):

        cont+=1
        eeg, labels = _data 
        eeg, labels = eeg.to(device), labels.to(device)

        # Zero the gradients
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        # Perform forward pass to DSF
        outputs1 = model1(eeg)  # (batch, n_class)
        outputs1 = outputs1.squeeze(1)
        
        if   experiment_1 == 'exp1': # sin normalizacion 
            pass
        elif experiment_1 == 'exp2': # normalizacion por canal
            mean     = outputs1.mean(dim=2, keepdim=True)
            std      = outputs1.std(dim=2, keepdim=True)

            outputs1 = (outputs1 - mean) / std

        elif experiment_1 == 'exp3': # normalizacion global
            outputs1 = (outputs1 - outputs1.mean()) / outputs1.std()
        
        outputs1 = outputs1.to('cpu')

        # create spectrogram from outputs1
        spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
        spectrograms = torch.from_numpy(spectrograms)

        spectrograms_transformed = transform_train(spectrograms) 

        # contact tensors
        spectrograms2train = torch.cat((spectrograms, spectrograms_transformed), axis=0)
        spectrograms2train = spectrograms2train.to(device)

        # las labels tb se duplican
        labels2train       = torch.cat((labels, labels), axis=0)

        # Perform forward pass to iESPnet
        outputs2 = model2(spectrograms2train)

        m     = nn.Sigmoid()
        probs = m(outputs2)
        
        y_true  = torch.max(labels2train, dim = 1)[0]
        y_pred  = torch.max(probs, dim = 1)[0]
        
        if cont == 1:
            Y_true = y_true
            Y_pred = y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)


        # Compute loss
        loss = criterion(outputs2, labels2train)

        # Perform backward pass
        loss.backward()
        train_loss += loss.item()

        # Perform optimization
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        
        # record training loss
        train_losses.append(loss.item())

        del _data
        torch.cuda.empty_cache()

    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()

    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr = average_precision_score(y_val_true,val_pred)
        
    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))
    return train_losses, train_aucpr

def training_dsf_iespnet_abl(model1, model2 ,device, train_loader, transform_train, criterion, optimizer1, optimizer2, scheduler1, scheduler2, epoch):  
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    top_db           = 60.0

    train_loss = 0.0

    # train with early stopping to track the training loss as the model trains
    train_losses = []
    # precision = Precision(average=False, device=device)
    # recall    = Recall(average=False, device=device)

    cont = 0

    model1.train()
    model2.train()

    #train_loader es un dataloader que devuelve el eeg y el label continuo suavizado, es decir con el smoothing_label
    for batch_idx, _data in enumerate(train_loader):

        cont+=1
        eeg, labels = _data 
        eeg, labels = eeg.to(device), labels.to(device)

        # Zero the gradients
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        # Perform forward pass to DSF
        outputs1 = model1(eeg)  # (batch, n_class)
        outputs1 = outputs1.squeeze(1)      
        outputs1 = (outputs1 - outputs1.mean()) / outputs1.std() # normalizacion global
        outputs1 = outputs1.to('cpu')

        # create spectrogram from outputs1
        spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
        spectrograms = torch.from_numpy(spectrograms)

        spectrograms_transformed = transform_train(spectrograms) 

        # contact tensors
        spectrograms2train = torch.cat((spectrograms, spectrograms_transformed), axis=0)
        spectrograms2train = spectrograms2train.to(device)

        # las labels tb se duplican
        labels2train       = torch.cat((labels, labels), axis=0)

        # Perform forward pass to iESPnet
        outputs2 = model2(spectrograms2train)

        m     = nn.Sigmoid()
        probs = m(outputs2)
        
        y_true  = torch.max(labels2train, dim = 1)[0]
        y_pred  = torch.max(probs, dim = 1)[0]
        
        if cont == 1:
            Y_true = y_true
            Y_pred = y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)


        # Compute loss
        loss = criterion(outputs2, labels2train)

        # Perform backward pass
        loss.backward()
        train_loss += loss.item()

        # Perform optimization
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        
        # record training loss
        train_losses.append(loss.item())

        del _data
        torch.cuda.empty_cache()

    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()

    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr = average_precision_score(y_val_true,val_pred)
        
    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))
    return train_losses, train_aucpr

def training_dsf_iespnet_abl_no_sche(model1, model2 ,device, train_loader, transform_train, criterion, optimizer1, optimizer2, scheduler2, epoch):  
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    top_db           = 60.0

    train_loss = 0.0

    # train with early stopping to track the training loss as the model trains
    train_losses = []
    # precision = Precision(average=False, device=device)
    # recall    = Recall(average=False, device=device)

    cont = 0

    model1.train()
    model2.train()

    #train_loader es un dataloader que devuelve el eeg y el label continuo suavizado, es decir con el smoothing_label
    for batch_idx, _data in enumerate(train_loader):

        cont+=1
        eeg, labels = _data 
        eeg, labels = eeg.to(device), labels.to(device)

        # Zero the gradients
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        # Perform forward pass to DSF
        outputs1 = model1(eeg)  # (batch, n_class)
        outputs1 = outputs1.squeeze(1)      
        outputs1 = (outputs1 - outputs1.mean()) / outputs1.std() # normalizacion global
        outputs1 = outputs1.to('cpu')

        # create spectrogram from outputs1
        spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
        spectrograms = torch.from_numpy(spectrograms)

        spectrograms_transformed = transform_train(spectrograms) 

        # contact tensors
        spectrograms2train = torch.cat((spectrograms, spectrograms_transformed), axis=0)
        spectrograms2train = spectrograms2train.to(device)

        # las labels tb se duplican
        labels2train       = torch.cat((labels, labels), axis=0)

        # Perform forward pass to iESPnet
        outputs2 = model2(spectrograms2train)

        m     = nn.Sigmoid()
        probs = m(outputs2)
        
        y_true  = torch.max(labels2train, dim = 1)[0]
        y_pred  = torch.max(probs, dim = 1)[0]
        
        if cont == 1:
            Y_true = y_true
            Y_pred = y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)


        # Compute loss
        loss = criterion(outputs2, labels2train)

        # Perform backward pass
        loss.backward()
        train_loss += loss.item()

        # Perform optimization
        optimizer1.step()
        optimizer2.step()
        scheduler2.step()
        
        # record training loss
        train_losses.append(loss.item())

        del _data
        torch.cuda.empty_cache()

    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()

    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr = average_precision_score(y_val_true,val_pred)
        
    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))
    return train_losses, train_aucpr

def training_dsf_iespnet_abl_early(model1, model2 ,device, train_loader, transform_train, criterion, optimizer1, optimizer2, scheduler1, scheduler2, epoch):  
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    top_db           = 60.0

    train_loss = 0.0

    # train with early stopping to track the training loss as the model trains
    train_losses = []
    # precision = Precision(average=False, device=device)
    # recall    = Recall(average=False, device=device)

    cont = 0

    model1.train()
    model2.train()

    #train_loader es un dataloader que devuelve el eeg y el label continuo suavizado, es decir con el smoothing_label
    for batch_idx, _data in enumerate(train_loader):

        cont+=1
        eeg, labels = _data 
        eeg, labels = eeg.to(device), labels.to(device)

        # Zero the gradients
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        # Perform forward pass to DSF
        outputs1 = model1(eeg)  # (batch, n_class)
        outputs1 = outputs1.squeeze(1)
        outputs1 = (outputs1 - outputs1.mean()) / outputs1.std() # normalizacion global
        outputs1 = outputs1.to('cpu')

        # create spectrogram from outputs1
        spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
        spectrograms = torch.from_numpy(spectrograms)

        spectrograms_transformed = transform_train(spectrograms) 

        # contact tensors
        spectrograms2train = torch.cat((spectrograms, spectrograms_transformed), axis=0)
        spectrograms2train = spectrograms2train.to(device)

        # las labels tb se duplican
        labels2train       = torch.cat((labels, labels), axis=0)

        # Perform forward pass to iESPnet
        outputs2 = model2(spectrograms2train)

        m     = nn.Sigmoid()
        probs = m(outputs2)
        
        y_true  = torch.max(labels2train, dim = 1)[0]
        y_pred  = torch.max(probs, dim = 1)[0]
        
        if cont == 1:
            Y_true = y_true
            Y_pred = y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)


        # Compute loss
        loss = criterion(outputs2, labels2train)

        # Perform backward pass
        loss.backward()
        train_loss += loss.item()

        # Perform optimization
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        
        # record training loss
        train_losses.append(loss.item())

        del _data
        torch.cuda.empty_cache()

    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()

    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr = average_precision_score(y_val_true,val_pred)
        
    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))
    return train_losses, train_aucpr

def training_iespnet(model ,device, train_loader, criterion, optimizer, scheduler, epoch):  
    train_loss   = 0.0
    train_losses = []

    # precision = Precision(average=False, device=device)
    # recall    = Recall(average=False, device=device)

    cont = 0
    model.train()

    for batch_idx, _data in enumerate(train_loader):
        cont+=1
        
        spectrograms, labels = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Perform forward pass
        outputs = model(spectrograms)
        
        m       = nn.Sigmoid()
        probs   = m(outputs)
        
        y_true  = torch.max(labels, dim =1)[0]
        y_pred  = torch.max(probs, dim=1)[0]
        
        if cont == 1:
            Y_true = y_true
            Y_pred = y_pred

        else:                
            Y_true = torch.cat((Y_true, y_true), axis=0)
            Y_pred = torch.cat((Y_pred, y_pred), axis=0)

        # Compute loss
        loss = criterion(outputs, labels)

        # Perform backward pass
        loss.backward()
        train_loss += loss.item()

        # Perform optimization
        optimizer.step()
        scheduler.step()
        
        # record training loss
        train_losses.append(loss.item())
    
        del _data
        torch.cuda.empty_cache()

    y_val_true, val_pred = Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()

    if np.isnan(val_pred).any():
        print('nan found in pred')
        train_aucpr = 0
    else:   
        train_aucpr = average_precision_score(y_val_true,val_pred)

    print('Train Epoch: {} \tTrainLoss: {:.6f} \tTrainAUCpr: {:.6f}'.format(epoch, np.mean(train_losses), train_aucpr))
  
    return train_losses, train_aucpr
    

def validate(model, device, val_loader, criterion, epoch, figure_path):
    valid_losses = []

    cont = 0
    model.eval()
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            cont+=1
            # print(batch_idx)
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            m = nn.Sigmoid()
            probs = m(outputs)
            
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs.unsqueeze_(0)
            
            y_true  = torch.max(labels, dim =1)[0]
            y_pred  = torch.max(probs, dim=1)[0]
            
            if cont==1:
                Y_true = y_true
                Y_pred= y_pred

            else:                
                Y_true = torch.cat((Y_true, y_true), axis=0)
                Y_pred = torch.cat((Y_pred, y_pred), axis=0)
            
            
            loss = criterion(outputs, labels)
            valid_losses.append(loss.item())
            
            del spectrograms, loss
            torch.cuda.empty_cache()           
        
    valid_aucpr = average_precision_score(Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy())
    
    print('Train Epoch: {} \tValLoss:  {:.6f} \tValAUCpr: {:.6f}'.format(epoch, np.mean(valid_losses), valid_aucpr))
   
    return valid_losses, valid_aucpr

def validate_v2(model1, model2, device, val_loader, criterion, epoch, experiment_1, experiment_2):
    
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) 
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) 
    SPEC_NFFT        = 500
    if   experiment_2 == '.1':  
        top_db       = 40.0
    elif experiment_2 == '.2':
        top_db       = 60.0
    elif experiment_2 == '.3':
        top_db       = 80.0

    valid_losses = []
    cont         = 0

    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        for eeg, labels in val_loader:
            cont+=1
        
            eeg, labels = eeg.to(device), labels.to(device)
            outputs1    = model1(eeg)
            outputs1    = outputs1.squeeze(1)

            if   experiment_1 == 'exp1': # sin normalizacion 
                pass
            elif experiment_1 == 'exp2': # normalizacion por canal
                mean     = outputs1.mean(dim=2, keepdim=True)
                std      = outputs1.std(dim=2, keepdim=True)

                outputs1 = (outputs1 - mean) / std

            elif experiment_1 == 'exp3': # normalizacion global
                outputs1 = (outputs1 - outputs1.mean()) / outputs1.std()

            outputs1     = outputs1.to('cpu')

            # create spectrogram from outputs1
            spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
            spectrograms = torch.from_numpy(spectrograms)
            spectrograms = spectrograms.to(device)

            outputs2 = model2(spectrograms)

            m = nn.Sigmoid()
            probs = m(outputs2)
            
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs2.unsqueeze_(0)
            
            y_true  = torch.max(labels, dim =1)[0]
            y_pred  = torch.max(probs, dim=1)[0]
            
            if cont == 1:
                Y_true = y_true
                Y_pred= y_pred

            else:                
                Y_true = torch.cat((Y_true, y_true), axis=0)
                Y_pred = torch.cat((Y_pred, y_pred), axis=0)
            
            
            loss = criterion(outputs2, labels)
            valid_losses.append(loss.item())
            
            del eeg, loss
            torch.cuda.empty_cache()           
        
    valid_aucpr = average_precision_score(Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy())
    
    print('Train Epoch: {} \tValLoss:  {:.6f} \tValAUCpr: {:.6f}'.format(epoch, np.mean(valid_losses), valid_aucpr))
   
    return valid_losses, valid_aucpr

def validate_abl(model1, model2, device, val_loader, criterion, epoch):
    # create spectrogram    
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) 
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) 
    SPEC_NFFT        = 500
    top_db           = 60.0

    valid_losses = []
    cont         = 0

    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        for eeg, labels in val_loader:
            cont+=1
        
            eeg, labels = eeg.to(device), labels.to(device)

            outputs1    = model1(eeg)
            outputs1    = outputs1.squeeze(1)
            outputs1    = (outputs1 - outputs1.mean()) / outputs1.std() # normalizacion global
            outputs1    = outputs1.to('cpu')

            # create spectrogram from outputs1
            spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
            spectrograms = torch.from_numpy(spectrograms)
            spectrograms = spectrograms.to(device)

            outputs2 = model2(spectrograms)

            m = nn.Sigmoid()
            probs = m(outputs2)
            
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs2.unsqueeze_(0)
            
            y_true  = torch.max(labels, dim =1)[0]
            y_pred  = torch.max(probs, dim=1)[0]
            
            if cont == 1:
                Y_true = y_true
                Y_pred= y_pred

            else:                
                Y_true = torch.cat((Y_true, y_true), axis=0)
                Y_pred = torch.cat((Y_pred, y_pred), axis=0)
            
            
            loss = criterion(outputs2, labels)
            valid_losses.append(loss.item())
            
            del eeg, loss
            torch.cuda.empty_cache()           
        
    valid_aucpr = average_precision_score(Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy())
    
    print('Train Epoch: {} \tValLoss:  {:.6f} \tValAUCpr: {:.6f}'.format(epoch, np.mean(valid_losses), valid_aucpr))
   
    return valid_losses, valid_aucpr

def validate_iespnet(model, device, val_loader, criterion, epoch):
    valid_losses = []

    cont = 0
    model.eval()
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            cont+=1
            
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            outputs = model(spectrograms)

            m     = nn.Sigmoid()
            probs = m(outputs)
            
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs.unsqueeze_(0)
            
            y_true  = torch.max(labels, dim =1)[0]
            y_pred  = torch.max(probs, dim=1)[0]
            
            if cont == 1:
                Y_true = y_true
                Y_pred = y_pred

            else:                
                Y_true = torch.cat((Y_true, y_true), axis=0)
                Y_pred = torch.cat((Y_pred, y_pred), axis=0)
            
            loss = criterion(outputs, labels)
            valid_losses.append(loss.item())
            
            del spectrograms, loss
            torch.cuda.empty_cache()           
        
    valid_aucpr = average_precision_score(Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy())
    
    print('Train Epoch: {} \tValLoss:  {:.6f} \tValAUCpr: {:.6f}'.format(epoch, np.mean(valid_losses), valid_aucpr))
   
    return valid_losses, valid_aucpr

def test_model(model, hparams, model_path, test_data):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))
     # following pytorch suggestion
    torch.backends.cudnn.benchmark = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    
    # load model
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
  
    test_loader = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False,
                            **kwargs)
    
    outputs = get_prediction(model, device, test_loader)
    # Process is completed.
    print('Testing process has finished.')
    return outputs

def test_model_v2(model1, model2, hparams, model_path, test_data, experiment_1, experiment_2):
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    
    # load model
    checkpoint = torch.load(model_path)

    model1.load_state_dict(checkpoint['model_state_dict1'])
    model2.load_state_dict(checkpoint['model_state_dict2'])
  
    test_loader = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False,**kwargs)
    outputs     = get_prediction_v2(model1, model2, device, test_loader, experiment_1, experiment_2)
    # Process is completed.
    print('Testing process has finished.')
    return outputs

def test_model_dsf_iespnet(model1, model2, hparams, model_path, test_data):
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    
    # load model
    checkpoint = torch.load(model_path)

    model1.load_state_dict(checkpoint['model_state_dict1'])
    model2.load_state_dict(checkpoint['model_state_dict2'])
  
    test_loader = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle = False,**kwargs)
    outputs     = get_prediction_abl(model1, model2, device, test_loader)
    # Process is completed.
    print('Testing process has finished.')
    return outputs

def test_model_dsf_iespnet_abl(model1, model2, hparams, model_path, test_data):
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    torch.backends.cudnn.benchmark     = False # reproducibilidad
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    
    # load model
    checkpoint = torch.load(model_path)

    model1.load_state_dict(checkpoint['model_state_dict1'])
    model2.load_state_dict(checkpoint['model_state_dict2'])
  
    test_loader = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False,**kwargs)
    outputs     = get_prediction_abl(model1, model2, device, test_loader)
    # Process is completed.
    print('Testing process has finished.')
    return outputs

def test_model_iespnet(model, hparams, model_path, test_data):
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))

    # following pytorch suggestion
    torch.backends.cudnn.benchmark = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    
    # load model
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
  
    test_loader = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle = False,**kwargs)
    outputs     = get_prediction_iespnet(model, device, test_loader)
    # Process is completed.
    print('Testing process has finished.')
    return outputs

def test_model_adaBatch(model, hparams, model_path, test_data, loops=10):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using {} device'.format(device))
     # following pytorch suggestion
    torch.backends.cudnn.benchmark = True

    kwargs = {'num_workers': hparams["num_workers"], 'pin_memory': True} if use_cuda else {}
    
    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
  
    test_loader = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False,
                            **kwargs)
    
    outputs = get_prediction_adaBatch(model, device, test_loader, loops)
    # Process is completed.
    print('Testing process has finished.')
    return outputs

def get_prediction_adaBatch(model, device, loader, loops=10):
    model.to(device)
    # first some forward passes to stimate the stats
    model.train()
    with torch.no_grad():
        for e in range(loops):
            for i, data_ in enumerate(loader):
                spectrograms, labels = data_
                
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
    

            
    model.eval()
    with torch.no_grad():
        for i, data_ in enumerate(loader):
            spectrograms, labels = data_
            
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            m = nn.Sigmoid()
            
            probs = m(outputs)
            
            # avoiding shape issues when the last batch has only one element
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs.unsqueeze_(0)
            
            if i==0:
                y_true = labels
                y_probs = probs
                y_outputs = outputs

            else:                
                y_true = torch.cat((y_true, labels), axis=0)
                y_probs = torch.cat((y_probs, probs), axis=0)
                y_outputs = torch.cat((y_outputs, outputs), axis=0)

     
    y_true = y_true.to('cpu').detach().numpy()
    y_probs = y_probs.to('cpu').detach().numpy()
    y_outputs = y_outputs.to('cpu').detach().numpy()
    
    l_true = np.zeros((len(y_true,)))
    idx_true = np.where(np.sum(y_true, axis=1)>0.0)[0].tolist()

    l_true[idx_true] = 1    
    t_true = np.argmax(y_true[idx_true], axis=1)
        
    
    prediction={'y_true': y_true,
                'y_prob': y_probs, 
                'y_output': y_outputs,
                't_true': t_true,
                'l_true': l_true
                }
    return prediction

def get_prediction(model, device, loader):
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        for i, data_ in enumerate(loader):
            spectrograms, labels = data_
            
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            
            m = nn.Sigmoid()
            
            probs = m(outputs)
            # print(len(probs.shape))
            # avoiding shape issues when the last batch has only one element
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs.unsqueeze_(0)
            if i==0:
                y_true = labels
                y_probs = probs
                y_outputs = outputs

            else:                
                y_true = torch.cat((y_true, labels), axis=0)
                y_probs = torch.cat((y_probs, probs), axis=0)
                y_outputs = torch.cat((y_outputs, outputs), axis=0)

    
    y_true = y_true.to('cpu').detach().numpy()
    y_probs = y_probs.to('cpu').detach().numpy()
    y_outputs = y_outputs.to('cpu').detach().numpy()
    
    l_true = np.zeros((len(y_true,)))
    idx_true = np.where(np.sum(y_true, axis=1)>0.0)[0].tolist()

    l_true[idx_true] = 1    
    t_true = np.argmax(y_true[idx_true], axis=1)
        
    
    prediction={'y_true': y_true,
                'y_prob': y_probs, 
                'y_output': y_outputs,
                't_true': t_true,
                'l_true': l_true
                }
    return prediction

def get_prediction_v2(model1, model2, device, loader, experiment_1, experiment_2):
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    if   experiment_2 == '.1':  
        top_db       = 40.0
    elif experiment_2 == '.2':
        top_db       = 60.0
    elif experiment_2 == '.3':
        top_db       = 80.0
    
    model1.to(device)
    model2.to(device)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        for i, data_ in enumerate(loader):
            eeg, labels = data_
            eeg, labels = eeg.to(device), labels.to(device)
            
            outputs1 = model1(eeg)
            outputs1 = outputs1.squeeze(1)

            if   experiment_1 == 'exp1': # sin normalizacion 
                pass
            elif experiment_1 == 'exp2': # normalizacion por canal
                mean     = outputs1.mean(dim=2, keepdim=True)
                std      = outputs1.std(dim=2, keepdim=True)

                outputs1 = (outputs1 - mean) / std

            elif experiment_1 == 'exp3': # normalizacion global
                outputs1 = (outputs1 - outputs1.mean()) / outputs1.std()

            outputs1 = outputs1.to('cpu')
           
            spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
            spectrograms = torch.from_numpy(spectrograms)
            spectrograms = spectrograms.to(device)
       
            outputs2 = model2(spectrograms)
         
            m     = nn.Sigmoid()
            probs = m(outputs2)

            # print(len(probs.shape))
            # avoiding shape issues when the last batch has only one element
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs2.unsqueeze_(0)
            if i==0:
                y_true = labels
                y_probs = probs
                y_outputs = outputs2

            else:                
                y_true    = torch.cat((y_true, labels), axis=0)
                y_probs   = torch.cat((y_probs, probs), axis=0)
                y_outputs = torch.cat((y_outputs, outputs2), axis=0)

    y_true    = y_true.to('cpu').detach().numpy()
    y_probs   = y_probs.to('cpu').detach().numpy()
    y_outputs = y_outputs.to('cpu').detach().numpy()

    l_true   = np.zeros((len(y_true,)))
    idx_true = np.where(np.sum(y_true, axis=1)>0.0)[0].tolist()

    l_true[idx_true] = 1    
    t_true           = np.argmax(y_true[idx_true], axis=1)


    prediction={
                'y_true': y_true,
                'y_prob': y_probs, 
                'y_output': y_outputs,
                't_true': t_true,
                'l_true': l_true
               }
    
    return prediction

def get_prediction_abl(model1, model2, device, loader):
    # create spectrogram
    ECOG_SAMPLE_RATE = 250
    ECOG_CHANNELS    = 4
    TT               = 1000 # window length
    SPEC_WIN_LEN     = int(ECOG_SAMPLE_RATE * TT / 1000 ) # win size
    overlap          = 500 
    SPEC_HOP_LEN     = int(ECOG_SAMPLE_RATE * (TT - overlap) / 1000) # Length of hop between windows.
    SPEC_NFFT        = 500  # to see changes in 0.5 reso
    top_db           = 60.0
    
    model1.to(device)
    model2.to(device)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        for i, data_ in enumerate(loader):
            eeg, labels = data_
            eeg, labels = eeg.to(device), labels.to(device)
            
            outputs1 = model1(eeg)
            outputs1 = outputs1.squeeze(1)
            outputs1 = (outputs1 - outputs1.mean()) / outputs1.std() # normalizacion global
            outputs1 = outputs1.to('cpu')
           
            spectrograms = get_spectrogram_2(outputs1, device, ECOG_SAMPLE_RATE, SPEC_NFFT, SPEC_WIN_LEN, SPEC_HOP_LEN, top_db)
            spectrograms = torch.from_numpy(spectrograms)
            spectrograms = spectrograms.to(device)
       
            outputs2 = model2(spectrograms)
         
            m     = nn.Sigmoid()
            probs = m(outputs2)

            # print(len(probs.shape))
            # avoiding shape issues when the last batch has only one element
            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs2.unsqueeze_(0)
            if i==0:
                y_true = labels
                y_probs = probs
                y_outputs = outputs2

            else:                
                y_true    = torch.cat((y_true, labels), axis=0)
                y_probs   = torch.cat((y_probs, probs), axis=0)
                y_outputs = torch.cat((y_outputs, outputs2), axis=0)

    y_true    = y_true.to('cpu').detach().numpy()
    y_probs   = y_probs.to('cpu').detach().numpy()
    y_outputs = y_outputs.to('cpu').detach().numpy()

    l_true   = np.zeros((len(y_true,)))
    idx_true = np.where(np.sum(y_true, axis=1)>0.0)[0].tolist()

    l_true[idx_true] = 1    
    t_true           = np.argmax(y_true[idx_true], axis=1)


    prediction={
                'y_true': y_true,
                'y_prob': y_probs, 
                'y_output': y_outputs,
                't_true': t_true,
                'l_true': l_true
               }
    
    return prediction

def get_prediction_iespnet(model, device, loader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, data_ in enumerate(loader):
            spectrograms, labels = data_
            
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            outputs = model(spectrograms)
            
            m     = nn.Sigmoid()
            probs = m(outputs)

            if len(probs.shape) == 1:
                probs.unsqueeze_(0)
                outputs.unsqueeze_(0)
            if i==0:
                y_true    = labels
                y_probs   = probs
                y_outputs = outputs

            else:                
                y_true    = torch.cat((y_true, labels), axis=0)
                y_probs   = torch.cat((y_probs, probs), axis=0)
                y_outputs = torch.cat((y_outputs, outputs), axis=0)

    
    y_true    = y_true.to('cpu').detach().numpy()
    y_probs   = y_probs.to('cpu').detach().numpy()
    y_outputs = y_outputs.to('cpu').detach().numpy()
    
    l_true    = np.zeros((len(y_true,)))
    idx_true  = np.where(np.sum(y_true, axis=1)>0.0)[0].tolist()

    l_true[idx_true] = 1    
    t_true           = np.argmax(y_true[idx_true], axis=1)
        
    
    prediction={
                'y_true': y_true,
                'y_prob': y_probs, 
                'y_output': y_outputs,
                't_true': t_true,
                'l_true': l_true
               }
    return prediction

def get_thr(rang, y_true, y_probs):
    # inputs should be of size n_samples x n_time
    MSE_=np.zeros((len(rang)))
    y_probstorch = torch.from_numpy(y_probs)

    for ii, r in enumerate(rang):
        n = torch.nn.Threshold(r, 0)
        y_pre_th = n(y_probstorch).numpy()
        
        # compute F1
        MSE_[ii]=mean_absolute_error(y_true, y_pre_th)
    idx_r = np.argmin(MSE_)
    return MSE_, rang[idx_r]

def get_F1max(rang, l_true, y_probs):
    F1_=np.zeros((len(rang)))
    
    for ii, r in enumerate(rang):
        probas_pred = np.max(y_probs, axis=1)
        l_predicted = np.zeros((len(l_true,)))
        idx_predicted = np.where(probas_pred>r)[0].tolist()
        l_predicted[idx_predicted] = 1
        # compute F1
        F1_[ii]=f1_score(l_true, l_predicted, average='macro')
    idx_r = np.argmax(F1_)
    return F1_, rang[idx_r]

def get_thr_output(rang, l_true, y_pre):
    F1_=np.zeros((len(rang)))
    y_pretorch = torch.from_numpy(y_pre)

    for ii, r in enumerate(rang):
        n = torch.nn.Threshold(r, 0)
        y_pre_th = n(y_pretorch).numpy()
        t_predicted = np.argmax(y_pre_th, axis=1) 

        l_predicted = np.zeros((len(l_true,)))
        idx_predicted = np.where(t_predicted!=0)[0].tolist()
        l_predicted[idx_predicted] = 1
        # compute F1
        F1_[ii]=f1_score(l_true, l_predicted, average='macro')
    
    idx_r = np.argmax(F1_)
    return F1_, rang[idx_r]

def get_performance_indices(y_true, y_prob, thr):
    
    # get true times and labels
    l_true = np.zeros((len(y_true,)))

    idx_true = np.where(np.sum(y_true, axis=1)>0.0)[0].tolist()
    l_true[idx_true] = 1    
    t_true = np.zeros((len(y_true,)))
    t_true[idx_true] = np.argmax(y_true[idx_true], axis=1)
    t_true = t_true.astype(int) #these are indices
    
    # post process the output given the threshold
    y_torch = torch.from_numpy(y_prob)
    n = torch.nn.Threshold(thr, 0)
    y_predicted = n(y_torch)
    
    l_predicted = np.zeros((len(y_predicted,)))
    idx_predicted = np.where(y_predicted.sum(dim=1)>0.0)[0].tolist()
    l_predicted[idx_predicted] = 1  
    t_predicted = np.zeros((len(y_predicted,)))
    t_predicted[idx_predicted] = y_predicted[idx_predicted].argmax(dim=1).numpy()
    t_predicted = t_predicted.astype(int)
    
    precision = precision_score(l_true, l_predicted, average='macro', zero_division=0)
    recall = recall_score(l_true, l_predicted, average='macro', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(l_true, l_predicted)
    epsilon = 1e-7
       
    f1 =  2* (precision*recall) / (precision + recall + epsilon)
    
    # MAE TIME
    # define the time vector used
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
    idx_tp = np.where((l_true==1) & (l_predicted==1))
   
    is_empty = np.size(idx_tp) == 0

    if is_empty:
        MAE_time = None
    else:
        MAE_time = mean_absolute_error(time[t_true[idx_tp]], time[t_predicted[idx_tp]])
            
    
    result={'accuracy': balanced_accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall, 
            'y_true': y_true,
            'y_pred': y_predicted, 
            't_true': t_true,
            't_pred': t_predicted,
            'l_true': l_true,
            'l_pred': l_predicted, 
            'proba': y_prob,
            'MAE_time': MAE_time
        }
    if MAE_time == None:
        print("Accuracy : {:.2f} F1 : {:.2f} Precission : {:.2f} Recall : {:.2f} ".format(balanced_accuracy, f1,precision,recall))
    else:
        print("Accuracy : {:.2f} F1 : {:.2f} Precission : {:.2f} Recall : {:.2f} MAEtime : {:.2f}".format(balanced_accuracy, f1,precision,recall, MAE_time))


    return result
