import csv
import os, sys
import os.path as osp
import time

import lmdb
from tqdm import tqdm
import pickle

import torch.utils.data as data
from torch.utils.data import DataLoader
import dataset
from pathlib import Path

def dumps_pickle(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    #return pa.serialize(obj).to_buffer() FutureWarning: 'pyarrow.deserialize' is deprecated as of 2.0.0
    return pickle.dumps(obj)

def dataset2lmdb(dataset, save_prefix, write_frequency=5000, max_num=400000, num_workers=16):

    dataloader = DataLoader(dataset,num_workers=num_workers,shuffle=True)


    if len(dataset) > max_num:
        lmdb_split = 0
        lmdb_path = "{}_{}.lmdb".format(save_prefix,lmdb_split)
        lmdb_split += 1
    else:
        lmdb_path = "{}.lmdb".format(save_prefix)

    if os.path.exists(lmdb_path):
        print("{} already exists".format(lmdb_path))
        exit(0)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 // 4, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []
    for idx, data in enumerate(dataloader):
        image, label, name = data[0].numpy(),data[1].numpy(),data[2][0]
        keys.append(u'{}'.format(name).encode('ascii'))
        txn.put(u'{}'.format(name).encode('ascii'), dumps_pickle((image, label)))
        if idx >0 and idx % max_num ==0:

            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps_pickle(keys))
                txn.put(b'__len__', dumps_pickle(len(keys)))
            print("Flushing database to {} ...".format(lmdb_path))
            db.sync()
            db.close()

            lmdb_path = "{}_{}.lmdb".format(save_prefix,lmdb_split)
            lmdb_split += 1
            isdir = os.path.isdir(lmdb_path)

            db = lmdb.open(lmdb_path, subdir=isdir,
                        map_size=1099511627776 // 18, readonly=False,
                        meminit=False, map_async=True)

            txn = db.begin(write=True)
            keys = []

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(dataloader)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

def folder2lmdb(manifest_path, lmdb_path, split="train", write_frequency=5000, sr=16000, max_num=200000, max_len=10, num_workers=5):
    from sys import getsizeof
    ds = dataset.FSD50KDataset3(manifest_path=manifest_path,split=split, max_len=max_len, sr=sr)
    #dataloader = DataLoader(ds,num_workers=num_workers,shuffle=True)
    lmdb_split = 0
    if len(ds) > max_num:
        lmdb_path = osp.join(lmdb_path, "{}_{}.lmdb".format(split,lmdb_split))
        lmdb_split += 1
    else:
        lmdb_path = osp.join(lmdb_path, "{}.lmdb".format(split))

    if os.path.exists(lmdb_path):
        print("{} already exists.".format(lmdb_path))
        exit(0)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []

    for idx, data in tqdm(enumerate(ds)):
        wavform, label, name = data[0].unsqueeze(0).numpy(),data[1].unsqueeze(0).numpy(),data[2]
        keys.append(u'{}'.format(name).encode('utf-8'))
        #before_serialize = getsizeof(wavform)
        dumps_obj = dumps_pickle((wavform, label)) # 这里没有放label，因为pretrain不需要label，hardcode成‘ALL’。在训练的时候，lmdb.py需要
        #after_serialize = getsizeof(dumps_obj)
        txn.put(u'{}'.format(name).encode('utf-8'), dumps_obj)

        if idx >0 and idx % max_num ==0:

            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps_pickle(keys))
                txn.put(b'__len__', dumps_pickle(len(keys)))
            print("Flushing database to {} ...".format(lmdb_path))
            db.sync()
            db.close()

            lmdb_path = osp.join(lmdb_path, "{}_{}.lmdb".format(split,lmdb_split))
            lmdb_split += 1
            isdir = os.path.isdir(lmdb_path)

            db = lmdb.open(lmdb_path, subdir=isdir,
                        map_size=1099511627776 // 18, readonly=False,
                        meminit=False, map_async=True)

            txn = db.begin(write=True)
            keys = []

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(ds)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return ds

# 这个函数是为了处理不等长的音乐，将音乐一首切割成定长的片段，需要MusicDataset实现cycleSeg的功能，返回一个list。
# 但是由于list非常消耗memory，write_frequency要设置到很小，才可以正常运行lmdb的写入功能。
# 不推荐采用这种方式写入，应当在此之前将音乐切分成定长的片段存储成wav/mp3，或者之后训练时随机选取定长的segment
def music2lmdb(data_path, split="train", sr=22050, stride=20, window_size=30, write_frequency=1000, max_num=200000, num_workers=5, transform=None):
    lmdb_path_ub = os.path.join(data_path, "lmdb_path_ub")
    ds = dataset.MusicSegmentDataset(data_path, split=split, sr=sr, stride=stride, window_size=window_size, transform=transform)
    ds_len = len(ds)*10  # 估算一下切割之后30s片段的个数，平均一首时间是3.55分钟，stride是20秒，平均一首就是10个片段。

    if ds_len > max_num:
        lmdb_split = 0
        lmdb_path = osp.join(lmdb_path_ub, "{}_{}.lmdb".format(split,lmdb_split))
        lmdb_split += 1
    else:
        lmdb_path = osp.join(lmdb_path_ub, "{}.lmdb".format(split))

    if os.path.exists(lmdb_path):
        print("{} already exists. Overwriting it.".format(lmdb_path))
        #exit(0)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 // 4, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []
    clip_idx = 0

    for _, ds_data in tqdm(enumerate(ds)):
        results, label = ds_data[0], ds_data[1].unsqueeze(0).numpy()
        for clip, file_name in results:
            spec, name = clip.unsqueeze(0).numpy(), file_name
            keys.append(u'{}'.format(name).encode('utf-8'))
            txn.put(u'{}'.format(name).encode('utf-8'), dumps_pickle((spec, label)))
            if clip_idx > 0 and clip_idx % max_num == 0:
                txn.commit()
                with db.begin(write=True) as txn:
                    txn.put(b'__keys__', dumps_pickle(keys))
                    txn.put(b'__len__', dumps_pickle(len(keys)))
                print("Flushing database to {} ...".format(lmdb_path))
                db.sync()
                db.close()

                lmdb_path = osp.join(lmdb_path_ub, "{}_{}.lmdb".format(split, lmdb_split))
                lmdb_split += 1
                isdir = os.path.isdir(lmdb_path)

                db = lmdb.open(lmdb_path, subdir=isdir,
                               map_size=1099511627776 // 4, readonly=False,
                               meminit=False, map_async=True)

                txn = db.begin(write=True)
                keys = []
            if clip_idx % write_frequency == 0:
                print("split segments(estimate): [%d/%d]" % (clip_idx, ds_len))
                txn.commit()
                txn = db.begin(write=True)
            clip_idx += 1
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))
    print("Flushing database ...")
    db.sync()
    db.close()

    return ds

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys= pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        waveform, label = unpacked
        if self.transform is not None:
            waveform = self.transform(waveform)


        return waveform, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'



if __name__ == "__main__":
    import sys
    path,split = sys.argv[1:]

    #folder2lmdb(path,split=split)
    #ds = dataset.FSD50KDataset3(path,split=split,multilabel=True)
    #dataset2lmdb(ds,save_prefix=os.path.join(path,split))
    folder2lmdb(path,split="train", lmdb_path="/root/autodl-tmp/audioset/lmdb_path_ub")

    """
    folder2lmdb(path,split="eval")
    folder2lmdb(path,split="train")
    """
    ds = ImageFolderLMDB(os.path.join(path,"train.lmdb"))
    i = iter(ds)
    next(i)
