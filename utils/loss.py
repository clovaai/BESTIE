"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import torch

    
class L1_Loss(torch.nn.Module):
    ''' L1 loss for Offset map (without Instance-aware Guidance)'''
    def __init__(self):
        super(L1_Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, out, target, weight):
        loss = self.l1_loss(out, target)
        
        return loss
    
    
class Weighted_L1_Loss(torch.nn.Module):
    ''' Weighted L1 loss for Offset map (with Instance-aware Guidance)'''
    def __init__(self):
        super(Weighted_L1_Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, out, target, weight):
        loss = self.l1_loss(out, target) * weight
        
        if weight.sum() > 0:
            loss = loss.sum() / (weight > 0).float().sum()
        else:
            loss = loss.sum() * 0
        
        return loss
    
    
class MSELoss(torch.nn.Module):
    ''' MSE loss for center map (without Instance-aware Guidance)'''
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, out, target, weight):
        
        loss = self.mse_loss(out, target)
        
        return loss
        

class Weighted_MSELoss(torch.nn.Module):
    ''' MSE loss for center map (with Instance-aware Guidance)'''
    def __init__(self):
        super(Weighted_MSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, out, target, weight):
        
        loss = self.mse_loss(out, target) * weight
        
        if weight.sum() > 0:
            loss = loss.sum() / (weight > 0).float().sum()
        else:
            loss = loss.sum() * 0
        
        return loss

    
class DeepLabCE(torch.nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=255, top_k_percent_pixels=0.2, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels):
        
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        
        return pixel_losses.mean()
    
    
class RegularCE(torch.nn.Module):
    """
    Regular cross entropy loss for semantic segmentation, support pixel-wise loss weight.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=255, weight=None):
        super(RegularCE, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels):
        pixel_losses = self.criterion(logits, labels)
        
        mask = (labels != self.ignore_label)

        if mask.sum() > 0:
            pixel_losses = pixel_losses.sum() / mask.sum()
        else:
            pixel_losses = pixel_losses.sum() * 0
        
        return pixel_losses

    
    
def _neg_loss(pred, gt, weight):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * weight
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * weight

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weight):
        return self.neg_loss(out, target, weight)
    
