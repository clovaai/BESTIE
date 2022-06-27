"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
import torch
import argparse
import os
import cv2
import time
import random
import pickle
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models import model_factory
from utils.LoadData import get_dataset
from utils.my_optim import WarmupPolyLR
from utils.loss import Weighted_L1_Loss, Weighted_MSELoss, DeepLabCE
from utils.utils import AverageMeter, get_ins_map, get_ins_map_with_point

import chainercv
from chainercv.datasets import VOCInstanceSegmentationDataset
from chainercv.evaluations import eval_instance_segmentation_voc

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def parse():

    parser = argparse.ArgumentParser(description='BESTIE pytorch implementation')
    parser.add_argument("--root_dir", type=str, default='', help='Root dir for the project')
    parser.add_argument('--sup', type=str, help='supervision source', choices=["cls", "point"])
    parser.add_argument("--dataset", type=str, default='voc', choices=["voc", "coco"])
    parser.add_argument("--backbone", type=str, default='resnet50', choices=["resnet50", "resnet101", "hrnet34", "hrnet48"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=416)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_iter", type=int, default=50000)
    parser.add_argument("--warm_iter", type=int, default=2000, help='warm-up iterations')
    parser.add_argument("--train_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--resume', default=None, type=str, help='weight restore')

    parser.add_argument('--save_folder', default='checkpoints/test1', help='Location to save checkpoint models')
    parser.add_argument('--print_freq', default=200, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_freq', default=10000, type=int, help='interval of save checkpoint models')
    parser.add_argument("--cur_iter", type=int, default=0, help='current training interations')
    
    parser.add_argument("--gamma", type=float, default=0.9, help='learning rate decay power')
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help='threshold for pseudo-label generation')
    parser.add_argument("--refine_thresh", type=float, default=0.3, help='threshold for refined-label generation')
    parser.add_argument("--kernel", type=int, default=41, help='kernel size for point extraction')
    parser.add_argument("--sigma", type=int, default=6, help='sigma of 2D gaussian kernel')
    parser.add_argument("--beta", type=float, default=3.0, help='parameter for center-clustering')
    parser.add_argument("--bn_momentum", type=float, default=0.01)
    parser.add_argument('--refine', type=str2bool, default=True, help='enable self-refinement.')
    parser.add_argument("--refine_iter", type=int, default=0, help='self-refinement running iteration')
    parser.add_argument("--seg_weight", type=float, default=1.0, help='loss weight for segmantic segmentation map')
    parser.add_argument("--center_weight", type=float, default=200.0, help='loss weight for center map')
    parser.add_argument("--offset_weight", type=float, default=0.01, help='loss weight for offset map')
    
    parser.add_argument('--val_freq', default=1000, type=int, help='interval of model validation')
    parser.add_argument("--val_thresh", type=float, default=0.1, help='threhsold for instance-groupping in validation phase')
    parser.add_argument("--val_kernel", type=int, default=41, help='kernsl size for point extraction in validation phase')
    parser.add_argument('--val_flip', type=str2bool, default=True, help='enable flip test-time augmentation in vadliation phase')
    parser.add_argument('--val_clean', type=str2bool, default=False, help='cleaning pseudo-labels using image-level labels')
    parser.add_argument('--val_ignore', type=str2bool, default=False, help='ignore')
    
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=int(os.environ["LOCAL_RANK"]))
    
    return parser.parse_args()

def print_func(string):
    if torch.distributed.get_rank() == 0:
        print(string)
    
def save_checkpoint(save_path, model):
    if torch.distributed.get_rank() == 0:
        print('\nSaving state: %s\n' % save_path)
        state = {
            'model': model.module.state_dict(),
        }
        torch.save(state, save_path)

    
def train():
    
    batch_time = AverageMeter()
    avg_total_loss = AverageMeter()
    avg_seg_loss = AverageMeter()
    avg_pseudo_center_loss = AverageMeter()
    avg_refine_center_loss = AverageMeter()
    avg_pseudo_offset_loss = AverageMeter()
    avg_refine_offset_loss = AverageMeter()

    best_AP = -1
    
    model.train()
    start = time.time()
    end = time.time()
    epoch = 0

    for cur_iter in range(1, args.train_iter+1):
        
        try:
            img, label, seg_map, center_map, offset_map, weight, point_list = next(data_iter)
        except Exception as e:
            print_func("   [LOADER ERROR] " + str(e))
            
            epoch += 1
            data_iter = iter(train_loader)
            img, label, seg_map, center_map, offset_map, weight, point_list = next(data_iter)
            
            end = time.time()
            batch_time.reset()
            avg_total_loss.reset()
            avg_seg_loss.reset()
            avg_pseudo_center_loss.reset()
            avg_refine_center_loss.reset()
            avg_pseudo_offset_loss.reset()
            avg_refine_offset_loss.reset()
            
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        seg_map = seg_map.to(device, non_blocking=True)
        center_map = center_map.to(device, non_blocking=True)
        offset_map = offset_map.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        
        run_refine = args.refine and (cur_iter > args.refine_iter)
        
        if run_refine:
            out, c_label = model(img, seg_map, label, point_list)
        else:
            out = model(img)
        
        seg_loss = criterion['seg'](out['seg'], seg_map) * args.seg_weight
        center_loss_1 = criterion['center'](out['center'], center_map, weight) * args.center_weight
        offset_loss_1 = criterion['offset'](out['offset'], offset_map, weight) * args.offset_weight

        center_loss_2 = center_loss_1
        offset_loss_2 = offset_loss_1
        
        if run_refine and args.sup == 'cls':
            center_loss_2 = criterion['center'](out['center'], c_label['center'], 
                                                      c_label['weight']) * args.center_weight
            
        if run_refine:
            offset_loss_2 = criterion['offset'](out['offset'], c_label['offset'],
                                                      c_label['weight']) * args.offset_weight
        
        loss = seg_loss + (center_loss_1 + center_loss_2)*0.5 + (offset_loss_1 + offset_loss_2)*0.5
        
        # compute gradient and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
            
        batch_time.update((time.time() - end))
        end = time.time()
        
        avg_total_loss.update(loss.item(), img.size(0))
        avg_seg_loss.update(seg_loss.item(), img.size(0))
        avg_pseudo_center_loss.update(center_loss_1.item(), img.size(0))
        avg_refine_center_loss.update(center_loss_2.item(), img.size(0))
        avg_pseudo_offset_loss.update(offset_loss_1.item(), img.size(0))
        avg_refine_offset_loss.update(offset_loss_2.item(), img.size(0))
            
        if cur_iter % args.print_freq == 0:
            batch_time.synch(device)
            avg_total_loss.synch(device)
            avg_seg_loss.synch(device)
            avg_pseudo_center_loss.synch(device)
            avg_refine_center_loss.synch(device)
            avg_pseudo_offset_loss.synch(device)
            avg_refine_offset_loss.synch(device)
            
            if args.local_rank == 0:
                print('Progress: [{0}][{1}/{2}] ({3:.1f}%, {4:.1f} min) | '
                      'Time: {5:.1f} ms | '
                      'Left: {6:.1f} min | '
                      'TotalLoss: {7:.4f} | '
                      'SegLoss: {8:.4f} | '
                      'centerLoss: {9:.4f} ({10:.4f} + {11:.4f}) | '
                      'OffsetLoss: {12:.4f} ({13:.4f} + {14:.4f}) '.format(
                          epoch, cur_iter, args.train_iter, 
                          cur_iter/args.train_iter*100, (end-start) / 60,
                          batch_time.avg * 1000, (args.train_iter - cur_iter) * batch_time.avg / 60,
                          avg_total_loss.avg, avg_seg_loss.avg, 
                          avg_pseudo_center_loss.avg + avg_refine_center_loss.avg,
                          avg_pseudo_center_loss.avg, avg_refine_center_loss.avg,
                          avg_pseudo_offset_loss.avg + avg_refine_offset_loss.avg,
                          avg_pseudo_offset_loss.avg, avg_refine_offset_loss.avg,
                      )
                     )

        if args.local_rank == 0 and cur_iter % args.save_freq == 0:
            save_path = os.path.join(args.save_folder, 'last.pt')
            save_checkpoint(save_path, model)
            
        if cur_iter % args.val_freq == 0:
            val_score = validate()
            
            if args.local_rank == 0 and val_score['map'] > best_AP:
                best_AP = val_score['map']
                print('\n Best mAP50, iteration : %d, mAP50 : %.2f \n' % (cur_iter, best_AP))

                save_path = os.path.join(args.save_folder, 'best.pt')
                save_checkpoint(save_path, model)
                    
        end = time.time()
        
    if args.local_rank == 0:
        print('\n training done')
        save_path = os.path.join(args.save_folder, 'last.pt')
        save_checkpoint(save_path, model)
        
    val_score = validate()

    if args.local_rank == 0 and val_score['map'] > best_AP:
        best_AP = val_score['map']
        print('\n Best mAP50, iteration : %d, mAP50 : %.2f \n' % (cur_iter, best_AP))

        model_file = os.path.join(args.save_folder, 'best.pt')
        save_checkpoint(save_path, model)

        
def validate():
    model.eval()
    
    pred_seg_maps, pred_labels, pred_masks, pred_scores  = [], [], [], []
    val_dir = "val_temp_dir"
    if args.local_rank == 0:
        os.makedirs(val_dir, exist_ok=True)
    
    torch.distributed.barrier()
    for img, cls_label, points, fname, tsize in tqdm(valid_loader):
        target_size = int(tsize[0]), int(tsize[1])
        
        if args.val_flip:
            img = torch.cat( [img, img.flip(-1)] , dim=0)
            
        out = model(img.to(device), target_shape=target_size)
        
        if args.sup == 'point' and args.val_clean:
            pred_seg, pred_label, pred_mask, pred_score = get_ins_map_with_point(out, 
                                                                                 cls_label, 
                                                                                 points,
                                                                                 target_size, 
                                                                                 device,
                                                                                 args)
        else:
            pred_seg, pred_label, pred_mask, pred_score = get_ins_map(out, 
                                                                      cls_label, 
                                                                      target_size, 
                                                                      device,
                                                                      args)
            
        with open(f'{val_dir}/{fname[0]}.pickle', 'wb') as f:
            pickle.dump({
                'pred_label': pred_label,
                'pred_mask': pred_mask,
                'pred_score': pred_score,
            }, f)
        
    torch.distributed.barrier()
    
    ap_result = {"ap": None, "map": None}
    
    if args.local_rank == 0:
        pred_masks, pred_labels, pred_scores = [], [], []
        
        for fname in ins_gt_ids:
            with open(f'{val_dir}/{fname}.pickle', 'rb') as f:
                dat = pickle.load(f)
                pred_masks.append(dat['pred_mask'])
                pred_labels.append(dat['pred_label'])
                pred_scores.append(dat['pred_score'])
        
        ap_result = eval_instance_segmentation_voc(pred_masks, 
                                                   pred_labels, 
                                                   pred_scores,
                                                   ins_gt_masks, 
                                                   ins_gt_labels, 
                                                   iou_thresh=0.5)
        
        #print(ap_result)
        os.system(f"rm -rf {val_dir}")
        
    torch.distributed.barrier()
    
    model.train()
    
    return ap_result
        
            
if __name__ == '__main__':
    
    args = parse()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    torch.backends.cudnn.benchmark = True
    
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)

    # Init dirstributed system
    torch.distributed.init_process_group(
        backend="nccl", rank=args.local_rank, world_size=torch.cuda.device_count()
    )
    args.world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{args.gpu}")
        
    if args.local_rank == 0:
        os.makedirs(args.save_folder, exist_ok=True)

    """ load model """
    model = model_factory(args)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    
    # define loss function (criterion) and optimizer
    criterion = {"center" : Weighted_MSELoss(), 
                 "offset" : Weighted_L1_Loss(), 
                 "seg" : DeepLabCE()
                }
    
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            ckpt = torch.load(args.resume, map_location='cpu')['model']
            model.load_state_dict(new_dict, strict=True)
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))

        
    """ Get data loader """
    train_dataset = get_dataset(args, mode='train')
    valid_dataset = get_dataset(args, mode='val')
    print_func("number of train set = %d | valid set = %d" % (len(train_dataset), len(valid_dataset)))
    
    n_gpus = torch.cuda.device_count()
    batch_per_gpu = args.batch_size // n_gpus

    train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpus, rank=args.local_rank)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_per_gpu,
                              num_workers=args.num_workers, 
                              pin_memory=True, 
                              sampler=train_sampler, 
                              drop_last=True)
    
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=n_gpus, rank=args.local_rank)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=1,
                              num_workers=4, 
                              pin_memory=True, 
                              sampler=valid_sampler, 
                              drop_last=False)
    
    if args.train_epoch != 0:
        args.train_iter = args.train_epoch * len(train_loader)
    
    lr_scheduler = WarmupPolyLR(
        optimizer,
        args.train_iter,
        warmup_iters=args.warm_iter,
        power=args.gamma,
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.gpu])
    
    if args.val_freq != 0 and args.local_rank == 0:
        print("...Preparing GT dataset for evaluation")
        ins_dataset = VOCInstanceSegmentationDataset(split='val', data_dir=args.root_dir)

        ins_gt_ids = ins_dataset.ids
        ins_gt_masks = [ins_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(ins_dataset))]
        ins_gt_labels = [ins_dataset.get_example_by_keys(i, (2,))[0] for i in range(len(ins_dataset))]
    
    torch.distributed.barrier()
    print_func("...Training Start \n")
    print_func(args)
    
    #validate()
    train()
    