import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from audiossl.datasets import LMDBDataset
from transform import FrameATSTTrainTransform
import argparse
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class FrameATSTDataModule(LightningDataModule):
    def __init__(self,
                 train_data_path=None,
                 valid_data_path=None,
                 batch_size_per_gpu=256,
                 num_workers=10,
                 subset=200000,
                 win_length=1024,
                 aug_tea=True,
                 aug_stu=True,
                 freq_wrap=True,
                 mask_ratio=0.75,
                 mask_type="block",
                 sr=16000,
                 anchor_len=6.,
                 mask_len=5,
                 min_mask_len=2,
                 n_mels=64,
                 **kwargs,
                 ):
        super().__init__()
        self.transform = FrameATSTTrainTransform(
            sr=sr, win_length=win_length,
            aug_tea=aug_tea, aug_stu=aug_stu,
            freq_wrap=freq_wrap, mask_ratio=mask_ratio,
            anchor_len=anchor_len, mask_type=mask_type,
            mask_len=mask_len, min_mask_len=min_mask_len,
            n_mels=n_mels,
            **kwargs)

        self.train_dataset = self._load_ds(train_data_path, "train", subset, sr)
        self.val_dataset = self._load_ds(valid_data_path, "valid", subset, sr)
        self.batch_size=batch_size_per_gpu
        self.num_workers=num_workers
        self.save_hyperparameters()
    
    def _load_ds(self, data_path, split, subset, sr):
        lmdb_datasets = []
        for db_path in os.listdir(data_path):
            if db_path.endswith(".lmdb"):
                lmdb_ds = LMDBDataset(os.path.join(data_path, db_path), split=split, subset=subset, sr=sr, transform=self.transform)
                lmdb_datasets.append(lmdb_ds)
        return ConcatDataset(lmdb_datasets)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               sampler=None,
                               drop_last=True)
        #length = len(dataloader)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     sampler=None,
                                     drop_last=True)
        # length = len(dataloader)
        return dataloader
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FrameATSTData")
        parser.add_argument("--data_path", type=str, default=None, help="data path")
        parser.add_argument('--batch_size_per_gpu', default=256, type=int,
            help='Per-GPU batch-size : number of distinct samples loaded on one GPU.')
        parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
        parser.add_argument('--subset', default=200000, type=int, help='subset of training data')
        parser.add_argument('--win_length', default=1024, type=int, help='windown length')
        parser.add_argument('--aug_tea', default=True, type=bool_flag, help='whether to augment the view fed into teacher branch; if symmetric is True, this augmented view is fed into both teacher and student.')
        parser.add_argument('--aug_stu', default=True, type=bool_flag, help='whether to augment the view fed into teacher branch; if symmetric is True, this augmented view is fed into both teacher and student.')
        parser.add_argument('--freq_wrap', default=True, type=bool_flag, help='freq wraping or not')
        parser.add_argument('--anchor_len',default=6.,type=float,help="length of training samples")
        parser.add_argument('--mask_ratio',default=0.75,type=float,help="masking ratio")
        parser.add_argument('--mask_len',default=5,type=int,help="masking block length")
        parser.add_argument('--min_mask_len',default=2,type=int,help="minimum masking block length")
        parser.add_argument('--n_mels',default=64,type=int,help="number of mel channels")
        parser.add_argument('--mask_type',default="block",type=str,help="masking type: random or block")

        return parent_parser
