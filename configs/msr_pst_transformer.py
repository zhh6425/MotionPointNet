#!/usr/bin/env python
# coding=utf-8
import time

class Config:
    def __init__(self):
        self.dataset = "MSRAction3D"  # [msr, ntu60]
        self.data_path = 'data/MSR-Action3D/data'
        self.model = 'PSTTransformer'
        self.stage = 'finetune'  # pretrain/finetune
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
        # P4D
        self.radius = 0.3
        self.nsamples = 32
        self.spatial_stride = 32
        self.temporal_kernel_size = 3
        self.temporal_stride = 2
        # transformer
        self.dim = 80
        self.depth = 5
        self.heads = 2
        self.dim_head = 40
        self.mlp_dim = 160
        self.dropout1 = 0.0
        self.dropout2 = 0.0
        # training
        self.batch_size = 14
        self.epochs = 50
        self.workers = 10
        base_lr = 0.01
        self.lr = base_lr if not self.pretrain else 0.02 * base_lr
        self.momentum = 0.9
        self.scheduler = 'step'  # 'step/cos'
        self.weight_decay = 1e-4
        self.lr_milestones = [20, 30]
        self.lr_gamma = 0.1
        self.lr_warmup_epochs = 10
        self.eta_min = 0.0
        self.label_smoothing = 0.0
        self.sync = True
        # output
        self.print_freq = 10  # Multi-head Number
        self.output_dir = f'{self.stage}_{self.model}_{self.dataset}_{self.clip_len}_frames'
        self.model_init = 'pretrain/pretrain_PSTTransformer_MSRAction3D_24_frames/checkpoint/model_last.pth'
        self.resume = None
        self.start_epoch = 0

        # Others
        self.local_rank = 0
        self.seed = 0
