import argparse
import json

import yaml
from pytorch_lightning import Trainer
from model import FrameATSTLightningModule
from data import FrameATSTDataModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from argparse import ArgumentParser
import os
import torch
import gc



def main(dict_args):
    # args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    # args.spec_h = args.n_mels
    # dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")
    logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    model = FrameATSTLightningModule(**dict_args)
    data = FrameATSTDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            devices=args.nproc,
                            #precision=16,
                            max_steps=args.max_steps,
                            logger=[logger_tb,logger_wb],
                            callbacks=[ModelCheckpoint(dirpath=args.save_path,
                                                       every_n_epochs=10,
                                                       save_top_k=-1,
                                                       filename="checkpoint-{epoch:3d}",
                                                       save_last=True,
                                                       ),
                                       LearningRateMonitor(logging_interval="step")],
                            )
    last_ckpt = os.path.join(args.save_path,"last.ckpt")

    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)

def parseConfig(configFile):

    with open(configFile, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = ArgumentParser("FrameATST")
    parser.add_argument("--save_path",type=str, default='/20A021/projects/audiossl/savedir')
    parser.add_argument('--nproc', type=int,  default=2)
    parser.add_argument('--patch_h', type=int,  default=64)
    parser.add_argument('--patch_w', type=int,  default=4)
    parser = FrameATSTLightningModule.add_model_specific_args(parser)
    parser = FrameATSTDataModule.add_data_specific_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    # 用my_train_small.yaml override train_small.sh中的参数
    dict_args.update(parseConfig(configFile="my_train_small.yaml"))

    # 需要根据epoch和batch_size来计算跟step相关的参数
    batch_size = dict_args["nproc"] * dict_args["batch_size_per_gpu"]
    steps_per_epoch = dict_args['subset'] / batch_size
    dict_args['steps_per_epoch'] = steps_per_epoch
    # dict_args["learning_rate"] = dict_args["learning_rate"] * batch_size/256 按这个比例缩小lr，还是overfit，再调小lr
    dict_args["spec_h"] = dict_args["n_mels"]
    dict_args['max_steps'] = int(steps_per_epoch * dict_args['max_epochs'])
    dict_args['warmup_steps'] = int(steps_per_epoch * dict_args['warmup_epochs'])
    # 保存这些args到训练的文件夹下。
    with open(os.path.join(dict_args["save_path"], 'args.json'), 'w') as fp:
        json.dump(dict_args, fp)

    # train
    main(dict_args)

