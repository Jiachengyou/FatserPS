#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
import torch
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
from .yolo_head_teacher import YOLOXHeadTeacher

from yolox.utils import (
    get_local_rank,
)

class YOLOXPSTeacher(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        
        in_channels = [256, 512, 1024]
        self.depth = 1.00
        self.width = 1.00
        self.act = "silu"
        self.backbone_teacher = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        self.head_teacher = YOLOXHeadTeacher(1, self.width, in_channels=in_channels, act=self.act)
        
        self.teacher = YOLOX(self.backbone_teacher, self.head_teacher)
        # resume teacher
        
        stat_dict_path = './YOLOX_outputs/yolox_l_PS_detection/latest_ckpt.pth'
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        ckpt = torch.load(stat_dict_path, map_location=self.device)
        self.teacher.load_state_dict(ckpt["model"], strict=False)
        print("load teacher detection model !!!")

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        
        
        # teacher
        if self.training:
            with torch.no_grad():
                fpn_outs_teacher = self.teacher.backbone(x)
                reg_outputs = self.teacher.head(
                    fpn_outs_teacher, targets, x
                )
        
        
        # student  
        fpn_outs = self.backbone(x)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, reid_loss, teacher_loss, num_fg, reg_teacher_loss = self.head(
                fpn_outs, targets, x, reg_outputs
            )
            outputs = {
                "total_loss": loss + reg_teacher_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                'reid_loss': reid_loss,
                'reid_teacher_loss': teacher_loss,
                "num_fg": num_fg,
                "reg_teacher_loss": reg_teacher_loss,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
