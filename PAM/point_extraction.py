"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import os
import numpy as np
import torch
import argparse
import torch.nn as nn

from models.classifier import vgg16_pam
from utils.LoadData import test_data_loader

def get_arguments():
    parser = argparse.ArgumentParser(description='PAM pytorch implement')
    parser.add_argument("--root_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--dataset", type=str, default='voc')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=321)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument('--checkpoint', default='checkpoints/PAM/ckpt_15.pth', help='Location to save checkpoint file')
    parser.add_argument('--save_dir', default='Peak_points', help='save dir for peak points')
    parser.add_argument("--alpha", type=float, default=0.7, help='hyperparameter for PAM (controller)')
    parser.add_argument("--conf_thresh", type=float, default=0.1, help='peak threshold')

    return parser.parse_args()
    

def peak_extract(heat, kernel=5, K=25):
    B, C, H, W = heat.size()
    
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()
    
    peak = heat * keep
    
    topk_scores, topk_inds = torch.topk(peak.view(B, C, -1), K)

    topk_inds = topk_inds % (H * W)
    topk_ys   = (topk_inds / W).int().float()
    topk_xs   = (topk_inds % W).int().float()
    
    topk_scores = topk_scores[0].float().detach().cpu().numpy()
    topk_ys = topk_ys[0].int().detach().cpu().numpy()
    topk_xs = topk_xs[0].int().detach().cpu().numpy()
    
    return topk_scores, topk_ys, topk_xs


def smoothing(heat, kernel=3):
    pad = (kernel - 1) // 2
    heat = torch.nn.functional.avg_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    return heat

        
if __name__ == '__main__':
    args = get_arguments()
    os.makedirs(args.save_dir, exist_ok=True)

    model = vgg16_pam(alpha=args.alpha)

    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)

    model.eval()
    model.cuda()

    data_loader = test_data_loader(args)

    with torch.no_grad():

        for idx, (img, label, meta) in enumerate(data_loader):
            img_name = meta['img_name'][0]
            ori_W, ori_H = int(meta['ori_size'][0]), int(meta['ori_size'][1])
            print("[%03d/%03d] %s" % (idx, len(data_loader), img_name), end='\r')

            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)

            # flip TTA
            _img = torch.cat( [img, img.flip(-1)] , dim=0)
            _label = torch.cat( [label, label] , dim=0)

            _, cam = model(_img, _label, (ori_H, ori_W))

            cam = (cam[0:1] + cam[1:2].flip(-1)) / 2.
            cam = smoothing(cam)

            peak_conf, peak_y, peak_x = peak_extract(cam, kernel=15)

            #####################################################################

            img_name = img_name.split("/")[-1][:-4]
            label = label[0].cpu().detach().numpy()
            valid_label = np.nonzero(label)[0]

            with open(os.path.join(args.save_dir, "%s.txt" % img_name), 'w') as peak_txt:
                for l in valid_label:
                    for conf, x, y in zip(peak_conf[l], peak_x[l], peak_y[l]):
                        if conf < args.conf_thresh:
                            break

                        peak_txt.write("%d %d %d %.3f\n" % (x, y, l, conf))

