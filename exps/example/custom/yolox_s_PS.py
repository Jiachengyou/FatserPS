#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from yolox.exp import Exp as MyExp

import random
import os
import torch
import torch.distributed as dist
import torch.nn as nn


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

#         # Define yourself dataset path
#         self.data_dir = "../PSDT-main/ImageSet/PRW/"
#         self.train_ann = "train_pid.json"
#         self.val_ann = "test_pid.json"
#         self.image_dir_ps = '../PSDT-main/data/PRW-v16.04.20/frames/'
#         self.test_dataset = 'prw'
# #         self.name="../../data/CUHK-SYSU/Image/SSM/"
#         self.dataset = 'prw'
    
    
        self.data_dir = "../PSDT-main/ImageSet/CUHK/"
        self.train_ann = "train_pid_new.json"
        self.val_ann = "test_new.json"
#         self.name="../../data/CUHK-SYSU/Image/SSM/"

        self.image_dir_ps = '../PSDT-main/data/CUHK-SYSU/Image/SSM'
        self.test_dataset = 'cuhk'
#         self.name="../../data/CUHK-SYSU/Image/SSM/"
        self.dataset = 'cuhk'

        self.num_classes = 80

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        
        
        self.data_num_workers = 4
        self.eval_interval = 10
        self.print_interval = 100
        self.save_history_ckpt = True
        self.seed = None
        
        
        
        
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            image_dir_ps=self.image_dir_ps,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
        