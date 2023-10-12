from __future__ import print_function
import datetime
import argparse
import importlib
import os
import time
import sys
import logging
import numpy as np
import random
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import default_collate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from thop import profile

import utils
import scheduler

import datasets as Datasets
import models as Models

def get_parameter_number(net, shape):
    #total_num = sum(p.numel() for p in net.parameters())
    #trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    input = torch.randn(shape).to(net.device)
    flops, params = profile(net, inputs=(input, None), verbose=False)

    return "FLOPs: %.4fG" % (flops / 1000 ** 3), "Params: %.4fM" % (params / 1000 ** 2)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(args, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for it, (clip, target, video_idx) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        clip, target = clip.to(device), target.to(device)
        # model.clear_memory()
        output_dict = model(clip, video_idx)
        output = output_dict['logit']

        loss = criterion(output, target) + 0.1 * output_dict['pre_loss'] if not args.pretrain else output_dict['pre_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_step = epoch * len(data_loader) + it
        batch_size = clip.shape[0]

        writer.add_scalar('train/training_loss', loss, cur_step)
        writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], cur_step)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))

        if args.pretrain:
            temporal_similarity= output_dict['temporal_similarity']
            target_similarity = output_dict['target_similarity']
            acc1, acc5 = utils.accuracy(temporal_similarity, target_similarity, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        writer.add_scalar('train/training_acc1', acc1, cur_step)
        writer.add_scalar('train/training_acc5', acc5, cur_step)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        lr_scheduler.step()
        sys.stdout.flush()

    return loss


def evaluate(model, criterion, data_loader, epoch, device, writer):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for it, (clip, target, video_idx) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # model.clear_memory()
            output_dict = model(clip, video_idx)
            output = output_dict['logit']
            loss = criterion(output, target) + 0.1 * output_dict['pre_loss']

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]

            cur_step = epoch * len(data_loader) + it

            writer.add_scalar('test/testing_loss', loss, cur_step)
            writer.add_scalar('test/testing_acc1', acc1, cur_step)
            writer.add_scalar('test/testing_acc5', acc5, cur_step)

            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if dist.get_rank() == 0:
        logging.info(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

    if dist.get_rank() == 0:
        logging.info(' * Video Acc@1 %f'%total_acc)
        logging.info(' * Class Acc@1 %s'%str(class_acc))
    writer.add_scalar('test/overall_acc', total_acc, epoch)

    return total_acc


def main(args):

    if args.output_dir:
        if 'finetune' in args.output_dir:
            log_path = os.path.join('OUTPUT', args.output_dir)
        elif 'pretrain' in args.output_dir:
            log_path = os.path.join('pretrain', args.output_dir)
        utils.mkdir(log_path)
        writer = SummaryWriter(log_dir=log_path)

    # logger
    log_dir = os.path.join(log_path, f'train_{args.dataset}.log')
    utils.setup_logger_dist(log_dir, name=args.dataset)

    if dist.get_rank() == 0:
        logging.info(args)
    set_random_seed(args.seed)

    device = torch.device('cuda')

    # Data loading code
    if dist.get_rank() == 0:
        logging.info("Loading data")

    st = time.time()

    DATASET = getattr(Datasets, args.dataset)
    dataset = DATASET(cfg=args.cfg_dataset,
                      root=args.data_path,
                      split='train')
                      # split='pretrain' if args.pretrain else 'train')
    dataset_test = DATASET(cfg=args.cfg_dataset,
                           root=args.data_path,
                           split='test')

    if dist.get_rank() == 0:
        logging.info("Creating data loaders")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if dist.get_rank() == 0:
        logging.info("Creating model")
    Model = getattr(Models, args.model)
    model = Model(args=args, num_classes=dataset.num_classes)

    if args.sync:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    if not args.pretrain and args.model_init:
        if dist.get_rank() == 0:
            logging.info(f"Loading pretrain model from {args.model_init}")

        # model_to_init = model
        model_dict = model.state_dict()

        checkpoint = torch.load(args.model_init, map_location='cpu')
        checkpoint = checkpoint['model']

        for key in list(checkpoint.keys()):
            if not key in list(model_dict.keys()):
                del checkpoint[key]

        missing_keys, unexp_keys = model.load_state_dict(checkpoint, strict=False)

        if dist.get_rank() == 0:
            logging.warning('Could not init from %s: %s', args.model_init, missing_keys)
            logging.warning('Unused keys in %s: %s', args.model_init, unexp_keys)

    in_shape = [1, args.clip_len, args.num_points, args.in_channels if hasattr(args, 'in_channels') else 3]
    params = get_parameter_number(model, in_shape)
    if dist.get_rank() == 0:
        logging.info(model)
        logging.info(params)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    lr = args.lr * torch.cuda.device_count()  # base lr times gpu num
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    if args.scheduler == 'cos':
        base_scheduler = scheduler.CosineLR(optimizer, eta_min=args.eta_min, num_epochs=args.epochs - args.lr_warmup_epochs,
                                            iters_per_epoch=len(data_loader))
        lr_scheduler = scheduler.Warmup(optimizer, base_scheduler, num_epochs=args.lr_warmup_epochs,
                                        iters_per_epoch=len(data_loader))
    elif args.scheduler == 'step':
        lr_scheduler = scheduler.WarmupMultiStepLR(optimizer, milestone_epochs=args.lr_milestones, gamma=args.lr_gamma, warmup_epochs=args.lr_warmup_epochs, iters_per_epoch=len(data_loader), warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:  # for testing or reload training
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if dist.get_rank() == 0:
        logging.info(f"Current stage is {args.stage}")
        logging.info("Start training")

    start_time = time.time()

    best_acc = 0
    min_loss = 1e5
    for epoch in range(args.start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        loss = train_one_epoch(args, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, writer)

        if not args.pretrain:
            acc = evaluate(model, criterion, data_loader_test, epoch, device, writer)

        if log_path:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            save_path = os.path.join(log_path, 'checkpoint')
            utils.mkdir(save_path)
            utils.save_on_master(
                checkpoint,
                os.path.join(save_path, 'model_last.pth'))
            if not args.pretrain:
                if acc > best_acc:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(save_path, 'model_best.pth'))
                    best_acc = acc
            else:
                if loss < min_loss:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(save_path, 'model_best.pth'))
                    min_loss = loss

        if dist.get_rank() == 0:
            if not args.pretrain:
                logging.info('Current best Acc {}'.format(best_acc))
            else:
                logging.info('Current loss {}'.format(loss))
                logging.info('Current min loss {}'.format(min_loss))

    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if dist.get_rank() == 0:
        logging.info('Training time {}'.format(total_time_str))
        logging.info('Accuracy {}'.format(best_acc))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--cfg", type=str, help="Training Config File")
    parser.add_argument("--local_rank", type=int, help="Local Rank")
    args = parser.parse_args()
    config_file = args.cfg.split('/')[-1].split('.')[0]
    config_setting = importlib.import_module("configs." + config_file).Config()
    config_setting.train_latent = args.pretrain
    config_setting.local_rank = int(os.environ["LOCAL_RANK"])  # args.local_rank
    return config_setting

if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(args)
