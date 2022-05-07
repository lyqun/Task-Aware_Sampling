import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np

from utils import Saver
from utils.config import load_config
from tools import parse
from utils.logger import setup_logger


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--cfg_path", type=str, default='./tasks/partnet_seg/configs/baseline.yaml')
parser.add_argument("--save_dir", type=str, default='logs/baseline')
parser.add_argument("--debug", action='store_true', default=False)

# for partnet segmentation only
parser.add_argument("-c", "--category", type=str, default=None)
parser.add_argument("-l", "--level", type=int, default=None)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--eval_workers', type=int, default=None)
parser.add_argument("--accum", type=int, default=24) # total batch size after gradient accumulation

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--dist", action='store_true', default=False)
parser.add_argument("--sync_bn", action='store_true', default=False)

args = parser.parse_args()



def train(model, optimizer, lr_scheduler, loss_func, train_loader, val_loader, train_metrics, val_metrics, cfgs, saver):
    for epoch in range(cfgs.train.max_epochs + 1):
        # train
        logger.info('====== epoch {} ======'.format(epoch))

        if args.dist:
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, loss_func, train_loader, train_metrics)

        if epoch == 2 and args.local_rank == 0:
            os.system('nvidia-smi')

        # update lr / loss configs
        if lr_scheduler is not None:
            lr_scheduler.step()
        loss_func.update(epoch)

        # evaluate
        if args.local_rank == 0 and epoch % cfgs.train.eval_epoch == 0:
            saver.save_checkpoint(model, epoch)

            logger.info(' ----- Evaluate ----- ')
            for metric in val_metrics:
                metric.eval_reset()
            
            model.eval()
            with torch.no_grad():
                for input_dict in val_loader:
                    pred_dict = model(input_dict)
                    for metric in val_metrics:
                        metric.eval_bs(input_dict, pred_dict)

            for metric in val_metrics:
                state, val = metric.update_best()
                if state:
                    saver.save_checkpoint(model, epoch, name='best_{}'.format(metric.name), best=True)
                    logger.info(' * {} (best): {:.4f}'.format(metric.name, val))
                else:
                    logger.info(' * {}: {:.4f}'.format(metric.name, val))
                

def train_one_epoch(model, optimizer, loss_func, train_loader, train_metrics):
    model.train()
    repeat = 1
    if args.accum > 0:
        repeat = args.accum // args.batch_size
    print_iter = max(1, len(train_loader) // 3)

    logger.info(' --- train, accumulate gradients for {} time(s). Total bacth size is {}.'.format(repeat, repeat * args.batch_size))

    loss_list = []
    loss_temp_list = []

    for metric in train_metrics:
        metric.eval_reset()
    
    optimizer.zero_grad()
    for it, input_dict in enumerate(train_loader):
        pred_dict = model(input_dict)
        for metric in train_metrics:
            metric.eval_bs(input_dict, pred_dict)

        loss = loss_func(input_dict, pred_dict)
        loss_norm = loss / repeat
        loss_norm.backward()

        # accumulate gradient
        if (it + 1) % repeat == 0 or (it + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_list.append(loss.item())
        loss_temp_list.append(loss.item())

        if (it + 1) % print_iter == 0:
            logger.info(' -- batch: {}/{} -- '.format(it + 1, len(train_loader)))
            logger.info('mean loss: {:.4f}'.format(np.mean(loss_temp_list)))
            loss_temp_list = []
    
    for metric in train_metrics:
        val = metric.eval_detach()
        logger.info(' * {}: {:.4f}'.format(metric.name, val))


if __name__ == '__main__':
    if torch.cuda.device_count() > 1 and args.dist:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    # load config
    cfgs = load_config(args.cfg_path)

    # for partnet segmentation only
    if args.category is not None and args.level is not None:
        cfgs.dataset.category = args.category
        cfgs.dataset.level = args.level
    
    if args.eval_workers is None:
        args.eval_workers = args.workers

    # set up logger
    logger = setup_logger(output=args.save_dir, distributed_rank=args.local_rank, name='train')
    logger.info(args)
    logger.info(cfgs)


    ## === parse config ===
    # build dataloader
    train_dst, val_dst, num_class = parse.parse_dataset(cfgs)
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)

    train_loader = DataLoader(train_dst, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), 
                            pin_memory=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=args.batch_size, shuffle=False,
                    pin_memory=True, num_workers=args.eval_workers)
    
    # build model
    model, optimizer, lr_scheduler = parse.parse_model(cfgs, num_class)
    model.cuda()
    if torch.cuda.device_count() > 1:
        if args.dist:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            if args.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = nn.DataParallel(model)

    # build loss function and evaluation metrics
    loss_func = parse.parse_loss(cfgs)
    train_metrics, val_metrics = parse.parse_metrics(cfgs)
    
    # train model
    saver = Saver(args.save_dir, max_files=5)
    train(model, optimizer, lr_scheduler, loss_func, train_loader, val_loader, train_metrics, val_metrics, cfgs, saver)
