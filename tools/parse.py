import torch
import importlib
from torch.optim.lr_scheduler import StepLR, MultiStepLR

def parse_dataset(cfgs, is_training=True, test_split='test'):
    # import dataset
    DATASET = importlib.import_module('tasks.{}.dataset'.format(cfgs.task.name))
    dst_cfgs = cfgs.dataset
    
    if not is_training:
        test_dst = DATASET.get_dataset(dst_cfgs, split=test_split, is_training=False)
        return test_dst, test_dst.num_class

    val_dst = DATASET.get_dataset(dst_cfgs, split='val')
    train_dst = DATASET.get_dataset(dst_cfgs, split='train')
    return train_dst, val_dst, train_dst.num_class


def parse_metrics(cfgs, is_training=True):
    METRIC = importlib.import_module('tasks.{}.metric'.format(cfgs.task.name))
    if not is_training:
        return METRIC.get_metrics(split='val')
    else:
        return METRIC.get_metrics(split='train'), \
            METRIC.get_metrics(split='val')


def parse_loss(cfgs):
    # import loss function
    LOSS = importlib.import_module('tasks.{}.loss'.format(cfgs.task.name))
    LOSS = LOSS.get_loss(cfgs.train.loss.name)
    return LOSS(cfgs.train.loss)


def parse_model(cfgs, num_class, is_training=True):
    # import model
    cfgs.model.num_class = num_class
    MODEL = importlib.import_module('models.{}.{}'.format(cfgs.model.backbone, cfgs.model.name))
    model = MODEL.get_model(cfgs.model)
    if not is_training:
        return model

    optim_cfg = cfgs.train.optimizer
    if optim_cfg.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=optim_cfg.lr,
            momentum=0.98, 
            weight_decay=optim_cfg.weight_decay, 
            nesterov=True)

        if optim_cfg.get('lr_scheduler'):
            if optim_cfg.lr_scheduler == 'step':
                lr_scheduler = StepLR(
                    optimizer, step_size=optim_cfg.lr_decay_steps, gamma=optim_cfg.lr_decay_rate)
            elif optim_cfg.lr_scheduler == 'multi-step':
                lr_scheduler = MultiStepLR(
                    optimizer, milestones=optim_cfg.lr_decay_list, gamma=optim_cfg.lr_decay_rate)
            else:
                raise NotImplementedError
        else:
            lr_scheduler = None
            '''
            def lr_step_decay(cur_epoch):
                cur_decay = 1
                for decay_step in optim_cfg.lr_decay_list:
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * optim_cfg.lr_decay
                return max(cur_decay, optim_cfg.lr_clip / optim_cfg.lr)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_step_decay)
            '''
    
    elif optim_cfg.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_cfg.lr)
        lr_scheduler = None
    else:
        raise NotImplementedError

    return model, optimizer, lr_scheduler
