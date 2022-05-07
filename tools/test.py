import torch
from torch.utils.data import DataLoader

import os
import argparse

from utils.config import load_config
from utils import load_checkpoint
from tools import parse


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--cfg_path", type=str, default='./tasks/partnet_seg/configs/baseline.yaml')
parser.add_argument("--save_dir", type=str, default='logs/baseline')
parser.add_argument("--split", type=str, default='test')
parser.add_argument("--debug", action='store_true', default=False)

parser.add_argument("--category", type=str, default=None)
parser.add_argument("--level", type=int, default=None)

parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--resume_epoch", type=int, default=None)
parser.add_argument("--resume_metric", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument('--workers', type=int, default=4)
args = parser.parse_args()


def log_str(msg):
    if args.debug:
        print(msg)
    else:
        with open(os.path.join(args.save_dir, 'test.log'), 'a+') as log_file:
            print(msg, file=log_file)

log_str(args)


def eval_one_epoch(model, val_loader, val_metrics):
    model.cuda().eval()
    log_str(' ----- Evaluate ----- ')
    for metric in val_metrics:
        metric.eval_reset()
    
    with torch.no_grad():
        for input_dict in val_loader:
            pred_dict = model(input_dict)
            for metric in val_metrics:
                metric.eval_bs(input_dict, pred_dict)

    for metric in val_metrics:
        val = metric.eval_detach()
        log_str(' * {}: {:.4f}'.format(metric.name, val))


if __name__ == '__main__':
    # load config
    cfgs = load_config(args.cfg_path)
    if args.category is not None and args.level is not None:
        cfgs.dataset.category = args.category
        cfgs.dataset.level = args.level

    # parse config
    val_dst, num_class = parse.parse_dataset(cfgs, is_training=False, test_split=args.split)
    val_metrics = parse.parse_metrics(cfgs, is_training=False)
    model = parse.parse_model(cfgs, num_class, is_training=False)

    # load checkpoint
    if args.resume_metric is not None:
        args.resume = os.path.join(args.save_dir, 'best_{}.pth'.format(args.resume_metric))
    elif args.resume_epoch is not None:
        args.resume = os.path.join(args.save_dir, 'checkpoint_epoch_{}.pth'.format(args.resume_epoch))
    load_checkpoint(model, args.resume)
    
    # initialize data loader
    val_loader = DataLoader(val_dst, batch_size=args.batch_size, shuffle=False,
                        pin_memory=True, num_workers=args.workers)

    # evaluate
    eval_one_epoch(model, val_loader, val_metrics)
