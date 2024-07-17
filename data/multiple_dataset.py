import random
import numpy as np
# from torch.utils.data.dataset import Dataset
from data.base_dataset import BaseDataset
from pdb import set_trace
from .arkit_dataset import ARKitDataset
from .biwi_dataset import BIWIDataset
from .w300lp_dataset import W300LPDataset

class MultipleDataset(BaseDataset):
    def __init__(self, opt):
        print(f'Load multiple dataset')
        # Multi-Dataset 부르기
        opt.csv_path_train = 'dataset/ARKitFace/ARKitFace_list/list/ARKitFace_train.csv'
        opt.random_sample = True
        opt.uniform_sample = False
        opt.img_size = 192
        dbs = []
        dbs.append(ARKitDataset(opt))
        dbs.append(W300LPDataset(opt))

        for i in range(len(dbs)):
            print(f'[{i}] {len(dbs[i])}')
        # train_dataset = MultipleDataset(dataloader, make_same_len=False)

        make_same_len = False
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]
