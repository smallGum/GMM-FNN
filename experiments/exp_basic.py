from torch.utils.data import DataLoader

from utils.tools import list_of_param_dicts
from utils.data_loader import Data_reader

class Exp_Basic(object):
    def __init__(self, param_dict, gpus=0):
        self.params = list_of_param_dicts(param_dict)
        self.gpus = gpus
        self.device = None
        self.model = None
        self.best_args = None

    def _build_model(self, args):
        raise NotImplementedError
    
    def _get_data(self, args, flag):
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        
        data_set = Data_reader(
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data,
            flag=flag,
            size=[args.hist_len, args.pred_len],
            target=args.target,
            inverse=args.inverse
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader
    
    def train(self):
        pass

    def retrain(self):
        pass

    def test(self):
        pass
