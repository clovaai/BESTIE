# ------------------------------------------------------------------------------
# Reference: https://github.com/qjadud1994/DRS/blob/main/scripts/train_cls.py
# ------------------------------------------------------------------------------
"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import os
import numpy as np
import torch
import argparse

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import vgg16_pam
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader
from utils.Metrics import Cls_Accuracy


def get_arguments():
    parser = argparse.ArgumentParser(description='PAM pytorch implement')
    parser.add_argument("--root_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--dataset", type=str, default='voc')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=321)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument('--save_folder', default='checkpoints/test', help='Location to save checkpoint models')
    parser.add_argument("--alpha", type=float, default=0.7, help='hyperparameter for PAM (controller)')

    return parser.parse_args()


def get_model(args):
    model = vgg16_pam(pretrained=True, alpha=args.alpha) 

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    
    optimizer = optim.SGD(
        [
            {'params': param_groups[0], 'lr': args.lr},
            {'params': param_groups[1], 'lr': 2*args.lr},
            {'params': param_groups[2], 'lr': 10*args.lr},
            {'params': param_groups[3], 'lr': 20*args.lr}
        ], 
        momentum=0.9, 
        weight_decay=args.weight_decay, 
        nesterov=True
    )

    return  model, optimizer


def train(current_epoch):
    global curr_iter
    losses = AverageMeter()
    cls_acc_metric = Cls_Accuracy()

    model.train()
    
    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for img, label, _ in train_loader:
        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)

        logit = model(img)

        """ classification loss """
        loss = F.multilabel_soft_margin_loss(logit, label)

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """ average meter """
        cls_acc_metric.update(logit, label)
        losses.update(loss.item(), img.size()[0])
        
        curr_iter += 1

        """ training log """
        if curr_iter % args.show_interval == 0:
            cls_acc = cls_acc_metric.compute_avg_acc()

            print('Epoch: [{}][{}/{}] '
                  'LR: {:.5f} '
                  'ACC: {:.5f} '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(
                    current_epoch, curr_iter%len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], cls_acc, loss=losses))

    
if __name__ == '__main__':
    args = get_arguments()
    
    n_gpu = torch.cuda.device_count()
    
    args.batch_size *= n_gpu
    args.num_workers *= n_gpu
    
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    train_loader = train_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)

    model, optimizer = get_model(args)

    curr_iter = 0
    for current_epoch in range(1, args.epoch+1):
        train(current_epoch)

        """ save checkpoint """
        if current_epoch % args.save_interval == 0 and current_epoch > 0:
            print('\nSaving state, epoch : %d \n' % current_epoch)
            state = {
                'model': model.module.state_dict(),
                #"optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': curr_iter,
            }
            model_file = args.save_folder + '/ckpt_' + repr(current_epoch) + '.pth'
            torch.save(state, model_file)
