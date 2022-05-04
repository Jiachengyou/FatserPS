#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        
        

        # Define yourself dataset path
        self.data_dir = "../TransPS-main/ImageSet/CUHK/"
        self.train_ann = "train_pid_new.json"
        self.val_ann = "test_new.json"
#         self.name="../../data/CUHK-SYSU/Image/SSM/"

        self.num_classes = 1

        self.max_epoch = 300
        self.data_num_workers = 4
        self.print_interval = 100
        self.eval_interval = 5
        