import numpy as np
import torch
from itertools import product as prod

"""
Function to convert dictionary of lists to list of dictionaries of all combinations of listed variables. 
Example:
    list_of_param_dicts({'a': [1, 2], 'b': [3, 4]}) ---> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
"""
def list_of_param_dicts(param_dict):
    """
    Arguments:
        param_dict   -(dict) dictionary of parameters
    """
    vals = list(prod(*[v for k, v in param_dict.items()]))
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
    
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + (std==0) * .001)
    
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * (std + (std==0) * .001)) + mean

class TriangularMask():
    def __init__(self, L, device="cpu"):
        mask_shape = [L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape), diagonal=1).to(device)
            self._mask[self._mask > 0.] = float('inf')

    @property
    def mask(self):
        return self._mask


class Args():
    """
    Arguments:
        arg_dict   -(dict) dictionary of model parameters
    """
    def __init__(self, arg_dict):
        self.__dict__ = arg_dict
    
    def __repr__(self):
        display = ""

        for k, v in self.__dict__.items():
            display += str(k) + ": " + str(v) + "\n"
        
        return display