import os
import pandas as pd
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

from utils.tools import StandardScaler

class Data_reader(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', data_name='ETT',
                 flag='train', size=None, target='OT', scale=True, inverse=False):
        # size [history_len, pred_len]
        if size == None:
            self.history_len = 24*7
            self.pred_len = 24
        else:
            self.history_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'val', 'test']
        self.type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = self.type_map[flag]

        self.target = target
        self.scale = scale
        self.inverse = inverse

        self.data_name = data_name
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.data_name == 'ETTh1':
            # train_set: 12 month
            # valid_set: 4 month
            # test_set:  4 month
            border1s = [0, 12*30*24 - self.history_len, 12*30*24+4*30*24 - self.history_len]
            border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        elif self.data_name == 'WTH':
            num_train = int(len(df_raw)*0.7)
            num_test = int(len(df_raw)*0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train-self.history_len, len(df_raw)-num_test-self.history_len]
            border2s = [num_train, num_train+num_vali, len(df_raw)]
        elif self.data_name == 'NASDAQ':
            # train_set: 90 days
            # valid_set: 7 days
            # test_set:  7 days
            border1s = [0, 13*30*90 - self.history_len, 13*30*90+13*30*7 - self.history_len]
            border2s = [13*30*90, 13*30*90+13*30*7, 13*30*90+13*30*14]
        else:
            raise Exception('Unknown dataset!')
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[self.type_map['train']]:border2s[self.type_map['train']]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.history_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

         
        