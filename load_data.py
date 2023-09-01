from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import h5py
import scipy.io as sio
import os

torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDataSet(Dataset):
    def __init__(self, images, labels, plabels=None, mode='train'):
        self.images = images
        self.labels = labels
        self.mode = mode
        if self.mode == 'train':
            self.plabels = plabels

    def __getitem__(self, index):
        if self.mode == 'train' and self.plabels is not None:
            return self.images[index], self.labels[index], self.plabels[index], index
        else:
            return self.images[index], self.labels[index], index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count

def get_loader(batch_size, db_name):
    base_path = './datasets/'
    path = base_path + db_name + '/mat/'

    # x: images  L:labels
    train_set = h5py.File(path + 'train.mat', 'r', libver='latest', swmr=True)
    train_L = np.array(train_set['L_tr'], dtype=np.float).T
    train_x = np.array(train_set['I_tr'], dtype=np.float).T
    # x_mean = np.mean(train_x, axis=0).reshape((1, -1))
    # train_x = train_x - x_mean
    train_P = np.array(train_set['PL_tr'], dtype=np.float).T # TODO!!!
    train_set.close()

    query_set = h5py.File(path + 'test.mat', 'r', libver='latest', swmr=True)
    query_L = np.array(query_set['L_te'], dtype=np.float).T
    query_x = np.array(query_set['I_te'], dtype=np.float).T
    # query_x = query_x - x_mean
    query_set.close()

    db_set = h5py.File(path + 'retrieval.mat', 'r', libver='latest', swmr=True)
    database_L = np.array(db_set['L_db'], dtype=np.float).T
    database_x = np.array(db_set['I_db'], dtype=np.float).T
    # database_x = database_x - x_mean
    db_set.close()

    imgs = {'train': train_x, 'query': query_x, 'database': database_x}
    labels = {'train': train_L, 'query': query_L, 'database': database_L}
    plabels = {'train': train_P, 'query': None, 'database': None}

    dataset = {x: CustomDataSet(images=imgs[x], labels=labels[x], plabels=plabels[x], mode=x)
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}
    # shuffle = {'query': False, 'train': False, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in
                  ['query', 'train', 'database']}

    return dataloader

def get_loader_web(batch_size, db_name, shuffle_mode = True):
    path = './datasets/web/'

    # x: images  L:labels
    if db_name == 'coco':
        train_set = h5py.File(path + 'coco80_web.mat', 'r', libver='latest', swmr=True)
    elif db_name == 'nus21':
        train_set = h5py.File(path + 'nus21_web.mat', 'r', libver='latest', swmr=True)
    elif db_name == 'nus10':
        train_set = h5py.File(path + 'nus10_web.mat', 'r', libver='latest', swmr=True)
    else:
        print('No web image can be used!')

    train_L = np.array(train_set['L_tr'], dtype=np.float).T   # L_tr: 8000*80 int64
    train_x = np.array(train_set['I_tr'], dtype=np.float).T   # I_tr: 8000*4096 single
    train_set.close()

    imgs = {'train': train_x}
    labels = {'train': train_L}

    dataset = {x: CustomDataSet(images=imgs[x], labels=labels[x]) for x in ['train']}

    # todo
    shuffle = {'train': shuffle_mode}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['train']}

    return dataloader
