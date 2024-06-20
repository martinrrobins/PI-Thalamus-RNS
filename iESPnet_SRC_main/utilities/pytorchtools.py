import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, epoch, val_loss, model, optimizer):

        score = -val_loss
        self.path_e = self.path + '_' + str(epoch) + '.pth'
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch,  val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            #always save the model
            self.save_checkpoint(epoch,  val_loss, model, optimizer)

        else:
            self.best_score = score
            self.save_checkpoint(epoch,  val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, epoch,  val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss from ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
            }, self.path_e)
        # torch.save(model.state_dict(), self.path_e)
        self.val_loss_min = val_loss
