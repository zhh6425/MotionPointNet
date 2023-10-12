#!/usr/bin/env python
# coding=utf-8
import time

class Config:
    def __init__(self):
        self.dataset = "MSRAction3D"  # [msr, ntu60]
        self.data_path = 'data/MSR-Action3D/data'
        self.model = 'MotionNet'
        self.stage = 'pretrain'  # pretrain/finetune
        self.pretrain = True if self.stage == 'pretrain' else False
        # input
        self.cfg_dataset = {
            'clip_len': 24,
            'frame_step': 1,
            'num_points': 2048,
            'data_meta': None,
            'cross_subject': True,
        }
        self.clip_len = self.cfg_dataset['clip_len']
        self.num_points = self.cfg_dataset['num_points']
        # PointNet2
        self.radius = [0.2, 0.4, 0.4]
        self.nsamples = [48, 32, 8]
        self.spatial_stride = [32, 8, 2]
        self.in_channels = 3
        self.mlps = [[64], [128, 256], [512, 1024]]
        self.ratio = 1.
        self.dropout = 0.2
        # training
        self.batch_size = 200
        self.epochs = 200
        self.workers = 16
        base_lr = 0.01
        self.lr = base_lr if not self.pretrain else 0.02 * base_lr
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.scheduler = 'cos'  # 'step/cos'
        self.lr_warmup_epochs = 5
        self.lr_milestones = [200, 250]
        self.lr_gamma = 0.1
        self.eta_min = 0.0
        self.label_smoothing = 0.1
        self.sync = True
        # output
        self.print_freq = 10
        self.output_dir = f'{self.stage}_{self.model}_{self.dataset}_{self.clip_len}_frames'
        self.model_init = None
        self.resume = None
        self.start_epoch = 0

        # Others
        self.local_rank = 0
        self.seed = 0
