import pickle

import lmdb
import numpy as np
import torch.utils.data as data
import os
import torch
import random
random.seed(1234)

from copy import deepcopy

class LMDBDataset(data.Dataset):
    def __init__(self, db_path, split="train", subset=None, transform=None, target_transform=None, return_key = False, sr=16000):
        self.db_path = db_path
        #lmdb_path = os.path.join(db_path, f"{split}.lmdb")
        self.return_key = return_key
        self.sr=sr
        self.subset = subset
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
            self.org_keys = deepcopy(self.keys)
            self.start = 0
            if subset is not None and subset < self.length:
                self.length = subset
                random.shuffle(self.keys)
                self.org_keys = deepcopy(self.keys)
                self.keys = self.keys[:subset]
                self.start = subset

        self.transform = transform
        self.target_transform = target_transform
        # with self.env.begin(write=False) as txn:
        #     byteflow = txn.get(self.keys[0])
        # unpacked = pickle.loads(byteflow)
        # self.num_classes = unpacked[1].shape[1]
        self.txn = self.env.begin(write=False)
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        key = self.keys[idx]
        byteflow = self.txn.get(self.keys[idx])
        unpacked = pickle.loads(byteflow)

        waveform, label = torch.from_numpy(unpacked[0].copy()).squeeze(0), torch.from_numpy(unpacked[1].copy()).squeeze(0)
        if self.transform is not None:
            transformed = self.transform(waveform)
            if self.target_transform is not None:
                transformed = list(transformed)
                transformed[0],label = self.target_transform(transformed[0],label)
                transformed = tuple(transformed)

            if self.return_key:
                return transformed, label, key
            else:
                return transformed, label
        else:
            if self.return_key:
                return waveform, label , key
            else:
                return waveform, label
    # def cycle(self):
    #     if self.start + self.subset > len(self.org_keys):
    #         self.keys = self.org_keys[self.start:] + self.org_keys[:self.start+self.subset - len(self.org_keys)]
    #         random.shuffle(self.org_keys)
    #         self.start = 0
    #         #self.start = self.start + self.subset - len(self.org_keys)
    #     else:
    #         self.keys = self.org_keys[self.start:self.start+self.subset]
    #         self.start = self.start+self.subset

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
