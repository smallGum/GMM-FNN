from experiments.exp_basic import Exp_Basic
from models.model import GMM_FNN
from utils.tools import EarlyStopping, Args, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import gc

import warnings
warnings.filterwarnings('ignore')

def neg_log_likelihood(mse_i, sigma_1k, pi_1k, device):
    # mse_i: (batch, )
    # sigma_1k: (k, )
    # pi_1k: (k, )
    eps = 1e-12
    sigma_1k = sigma_1k + eps
    c_pi = torch.tensor(np.pi).to(device)
    k_num = pi_1k.size(0)
    likelihood = 0.
    for i in range(k_num-1):
        cur_lh = pi_1k[i] * torch.exp(-mse_i/(2.*sigma_1k[i+1])) / torch.sqrt(sigma_1k[i+1]*2.*c_pi) 
        likelihood = likelihood + cur_lh
    cur_lh = pi_1k[k_num-1] * torch.exp(-mse_i/(2.*sigma_1k[0])) / torch.sqrt(sigma_1k[0]*2.*c_pi)
    likelihood = likelihood + cur_lh
    
    neg_log_lh = -torch.log(likelihood + eps)
    
    return neg_log_lh

def complete_data_log(mse_i, sigma_1k, pi_1k, device):
    # mse_i: (batch, )
    # sigma_1k: (k,)
    # pi_1k: (k, )
    
    c_pi = torch.tensor(np.pi).to(device)
    k_num = pi_1k.size(0)
    log_lh = 0.
    for i in range(k_num-1):
        cur_lh = pi_1k[i] * (torch.log(pi_1k[i]) - 0.5*torch.log(2.*c_pi) - 0.5*torch.log(sigma_1k[i+1]) - mse_i/(2.*sigma_1k[i+1]))
        log_lh = log_lh + cur_lh
    cur_lh = pi_1k[k_num-1] * (torch.log(pi_1k[k_num-1]) - 0.5*torch.log(2.*c_pi) - 0.5*torch.log(sigma_1k[0]) - mse_i/(2.*sigma_1k[0]))
    log_lh = log_lh + cur_lh
    
    neg_log_lh = -log_lh
    return neg_log_lh

class Exp_GMMFNN(Exp_Basic):
    def __init__(self, param_dict, gpus=0):
        super(Exp_GMMFNN, self).__init__(param_dict, gpus)
    
    def _build_model(self, args):
        if self.gpus is not None:
            if type(self.gpus) == list:
                self.device = torch.device('cuda:{}'.format(self.gpus[0]))
            else:
                self.device = torch.device('cuda:{}'.format(self.gpus))
        else:
            self.device = torch.device('cpu')
        
        model = GMM_FNN(
            args.hidden_units,
            args.dropout_rate,
            args.hist_len,
            args.pred_len,
            args.n_heads,
            args.d_ff,
            args.d_layers,
            self.device
        ).float()

        if self.gpus is not None and type(self.gpus) == list:
            model = nn.DataParallel(model, device_ids=self.gpus)
        else:
            model = model.to(self.device)
        
        return model
    
    def _select_optimizer(self, args):
        model_optim = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        return model_optim
    
    def _select_criterion(self, red=True):
        criterion =  nn.MSELoss(reduce=red)
        return criterion

    def _process_one_batch(self, args, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs, sigma, f_weights = self.model(batch_x)
        outputs = outputs.unsqueeze(2)
        if args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        return outputs.squeeze(), batch_y.squeeze(), sigma, f_weights
    
    def valid(self, args, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y) in enumerate(valid_loader):
            pred, true, _, _ = self._process_one_batch(args, valid_data, batch_x, batch_y)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss
    
    def train(self):
        print("Train with training data to find best arguments ......")

        best_val_loss = np.Inf
        data_name = Args(self.params[0]).data
        loss_fn = neg_log_likelihood if data_name=='NASDAQ' else complete_data_log
        for param in self.params:
            args = Args(param)

            train_data, train_loader = self._get_data(args, flag='train')
            valid_data, valid_loader = self._get_data(args, flag='val')
            test_data, test_loader = self._get_data(args, flag='test')
            self.model = self._build_model(args)

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)
            model_optim = self._select_optimizer(args)
            criterion =  self._select_criterion(False)
            valid_criterion = self._select_criterion()

            for epoch in range(args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x,batch_y) in enumerate(train_loader):
                    iter_count += 1

                    model_optim.zero_grad()
                    pred, true, sigma, f_weights = self._process_one_batch(args, train_data, batch_x, batch_y)

                    cent = criterion(pred, true)
                    sigma2 = torch.mean(sigma**2., dim=0)
                    loss = 0.0
                    for l in range(cent.size(1)):
                        loss = loss + loss_fn(cent[:, l], sigma2[0:l+1], f_weights[l, 0:l+1], self.device)
                    loss = torch.sum(loss)

                    train_loss.append(loss.item())

                    loss.backward()
                    model_optim.step()
                
                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                train_loss = np.average(train_loss)
                valid_loss = self.valid(args, valid_data, valid_loader, valid_criterion)
                test_loss = self.valid(args, test_data, test_loader, valid_criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                      epoch + 1, train_steps, train_loss, valid_loss, test_loss))
                
                early_stopping(valid_loss, epoch+1)
                if early_stopping.val_loss_min < best_val_loss:
                    best_val_loss = early_stopping.val_loss_min
                    self.best_args = args
                    self.best_args.train_epochs = early_stopping.best_epoch
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
                adjust_learning_rate(model_optim, epoch+1, args)
            
            del self.model
            gc.collect()
            self.model = None
            self.device = None

        print('Found best arguments: ')
        print(self.best_args)
    
    def retrain(self, setting):
        print("Retrain with training and validation data ......")
        
        train_data, train_loader = self._get_data(self.best_args, flag='train')
        valid_data, valid_loader = self._get_data(self.best_args, flag='val')
        self.model = self._build_model(self.best_args)

        train_steps = len(train_loader) + len(valid_loader)
        model_optim = self._select_optimizer(self.best_args)
        criterion =  self._select_criterion(False)
        loss_fn = neg_log_likelihood if self.best_args.data=='NASDAQ' else complete_data_log

        for epoch in range(self.best_args.train_epochs): 
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true, sigma, f_weights = self._process_one_batch(self.best_args, train_data, batch_x, batch_y)
                cent = criterion(pred, true)
                sigma2 = torch.mean(sigma**2., dim=0)
                loss = 0.0
                for l in range(cent.size(1)):
                    loss = loss + loss_fn(cent[:, l], sigma2[0:l+1], f_weights[l, 0:l+1], self.device)
                loss = torch.sum(loss)

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
        
            for i, (batch_x,batch_y) in enumerate(valid_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true, sigma, f_weights = self._process_one_batch(self.best_args, train_data, batch_x, batch_y)

                cent = criterion(pred, true)
                sigma2 = torch.mean(sigma**2., dim=0)
                loss = 0.0
                for l in range(cent.size(1)):
                    loss = loss + loss_fn(cent[:, l], sigma2[0:l+1], f_weights[l, 0:l+1], self.device)
                loss = torch.sum(loss)

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                  epoch + 1, train_steps, train_loss))
            
            adjust_learning_rate(model_optim, epoch+1, self.best_args)
        
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model.state_dict(), folder_path+'checkpoint.pth')
    
    def test(self, setting):
        test_data, test_loader = self._get_data(self.best_args, flag='test')

        self.model.eval()

        preds = []
        trues = []
        for i, (batch_x,batch_y) in enumerate(test_loader):
            pred, true, _, _ = self._process_one_batch(self.best_args, test_data, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        print('test shape:', preds.shape, trues.shape)

        mses = []
        maes = []
        rmses = []
        for i in range(preds.shape[1]):
            cur_pred = preds[:, i:i+1]
            cur_true = trues[:, i:i+1]
            cur_mae, cur_mse, cur_rmse, _, _, _, _ = metric(cur_pred, cur_true)
            mses.append(cur_mse)
            maes.append(cur_mae)
            rmses.append(cur_rmse)
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe, rse, nrmse  = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, nrmse]))
        np.save(folder_path+'task_metrics.npy', np.array([mses, maes, rmses]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
