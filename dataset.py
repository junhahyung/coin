import os
import re
import numpy as np
import torch
from attrdict import AttrDict
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler


DATASET_PATH = './data/dataset'


class CoinDataset(Dataset):
    def __init__(self, train, conf=None):
        self.conf = conf
        self.train = train
        
        # DATA READ
        DATA_PATH = DATASET_PATH + "/{}USDT_{}.csv".format(conf.pair, conf.intv)
        self.data = pd.read_csv(DATA_PATH)
        
        # Data Type
        if self.train:
            self.data = self.data[:int(len(self.data)*conf.ratio)]
        else:
            self.data = self.data[int(len(self.data)*conf.ratio):]
            
        # Normalization
        ohlcv = self.data.iloc[:,4:]
        ohlcv_norm = (2*(ohlcv-ohlcv.min())/(ohlcv.max()-ohlcv.min())-1)
        self.data_norm = pd.concat([self.data.iloc[:,:4], ohlcv_norm], axis = 1)

    def __len__(self):
        return len(self.data_norm)-self.conf.nhist-self.conf.ntarget+1

    def __getitem__(self, index):
        x_norm = self.data_norm.iloc[index:index+self.conf.nhist, 4:]   
        y_norm = self.data_norm.iloc[index+self.conf.nhist+self.conf.ntarget-1, 4:]
        x_orig = self.data.iloc[index:index+self.conf.nhist, 4:]
        y_orig = self.data.iloc[index+self.conf.nhist+self.conf.ntarget-1, 4:]
        isLong = x_orig.iloc[-1]['Close'] < y_orig['Close']
        if isLong:
            isLong = torch.ones(1).long()
        else:
            isLong = torch.zeros(1).long()

#         print(x_norm.shape, y_norm.shape, isLong, x_orig.shape, y_orig.shape)
#         print(x_norm.shape, y_norm.type, isLong, x_orig.shape, y_orig.shape)
        return \
            torch.FloatTensor(x_norm.values), \
            torch.FloatTensor(list(y_norm.values)), \
            isLong, \
            torch.FloatTensor(x_orig.values), \
            torch.FloatTensor(list(y_orig.values))

def get_dataset(conf):
    train_dataset = CoinDataset(True, conf)
    valid_dataset = CoinDataset(False, conf)

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                               shuffle=False, batch_size=conf.bs)  
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, sampler=valid_sampler,
                                               shuffle=False, batch_size=conf.bs)  

    return train_loader, valid_loader
