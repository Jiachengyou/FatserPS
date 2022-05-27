#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "../PSDT-main/ImageSet/CUHK/"
        self.train_ann = "train_pid_new.json"
        self.val_ann = "test_new.json"
#         self.name="../../data/CUHK-SYSU/Image/SSM/"

        self.num_classes = 1

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        
        
        self.data_num_workers = 4
        self.eval_interval = 10
        self.print_interval = 100
        self.save_history_ckpt = True
        self.seed = None
        