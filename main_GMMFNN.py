import argparse
import os
import torch

from experiments.exp_GMMFNN import Exp_GMMFNN
from utils.tools import list_of_param_dicts

exp_num = 1                          # experiment times
gpus = 0 if torch.cuda.is_available() else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="name of dataset",
                        type=str)
    parser.add_argument("hist_len",
                        help="length of history",
                        type=int)
    parser.add_argument("pred_len",
                        help="length of prediction",
                        type=int)
    sys_args = parser.parse_args()
    sys_data = sys_args.dataset
    sys_hist_len = sys_args.hist_len
    sys_pred_len = sys_args.pred_len

    targets = {'ETTh1': 'OT', 'NASDAQ': 'NDX', 'WTH': 'WetBulbCelsius'}
    filenames = {'ETTh1': 'ETTh1.csv', 'NASDAQ': 'nasdaq100.csv', 'WTH': 'WTH.csv'}

    param_dict = dict(
        model = ['GMM_FNN'],
        data = [sys_data],                       # dataset of experiment
        root_path = ['./data/'],                 # root path of the data file
        data_path = [filenames[sys_data]],       # data file
        target = [targets[sys_data]],            # target time series for UTSF

        hist_len = [sys_hist_len],               # input sequence length
        pred_len = [sys_pred_len],               # prediction sequence length

        dropout_rate = [0.1],
        hidden_units=[2048],
        n_heads = [8],
        d_ff = [1024],
        d_layers = [2],

        num_workers=[0],                 # data loader num workers
        train_epochs = [100],
        batch_size = [64],
        patience = [5],                  # early stopping patience
        learning_rate = [0.0001],
        lradj = ['type1'],
        inverse = [True]
    )

    params = list_of_param_dicts(param_dict)
    args = params[0]

    for ii in range(exp_num):
        # setting record of experiments
        setting = '{}_{}_hl{}_pl{}_{}'.format(args['model'], args['data'], args['hist_len'], args['pred_len'], ii)
        exp = Exp_GMMFNN(param_dict, gpus=gpus) # set experiments

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train()
        exp.retrain(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()


    