# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from utils.utils import refine_label_generation, refine_label_generation_with_point

################################################################################################

def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
               with_bn=True, with_relu=True):
    """convolution with bn and relu"""
    module = []
    has_bias = not with_bn
    module.append(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                  bias=has_bias)
    )
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


def depthwise_separable_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
                             with_bn=True, with_relu=True):
    """depthwise separable convolution with bn and relu"""
    del groups

    module = []
    module.extend([
        basic_conv(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes,
                   with_bn=True, with_relu=True),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
    ])
    if with_bn:
        module.append(nn.BatchNorm2d(out_planes))
    if with_relu:
        module.append(nn.ReLU())
    return nn.Sequential(*module)


def stacked_conv(in_planes, out_planes, kernel_size, num_stack, stride=1, padding=1, groups=1,
                 with_bn=True, with_relu=True, conv_type='basic_conv'):
    """stacked convolution with bn and relu"""
    if num_stack < 1:
        assert ValueError('`num_stack` has to be a positive integer.')
    if conv_type == 'basic_conv':
        conv = partial(basic_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=groups, with_bn=with_bn, with_relu=with_relu)
    elif conv_type == 'depthwise_separable_conv':
        conv = partial(depthwise_separable_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=1, with_bn=with_bn, with_relu=with_relu)
    else:
        raise ValueError('Unknown conv_type: {}'.format(conv_type))
    module = []
    module.append(conv(in_planes=in_planes))
    for n in range(1, num_stack):
        module.append(conv(in_planes=out_planes))
    return nn.Sequential(*module)

################################################################################################

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def set_image_pooling(self, pool_size=None):
        if pool_size is None:
            self.aspp_pooling[0] = nn.AdaptiveAvgPool2d(1)
        else:
            self.aspp_pooling[0] = nn.AvgPool2d(kernel_size=pool_size, stride=1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        # out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


################################################################################################

class SinglePanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, aspp_channels=None):
        super(SinglePanopticDeepLabDecoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.aspp(x)

        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        return x


class SinglePanopticDeepLabHead(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels[i],
                ),
                nn.Conv2d(head_channels[i], num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)
            
        return pred


class PanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, num_classes, instance_head_kwargs, **kwargs):
        super(PanopticDeepLabDecoder, self).__init__()
        
        self.semantic_decoder = SinglePanopticDeepLabDecoder(in_channels, feature_key, low_level_channels,
                                                             low_level_key, low_level_channels_project,
                                                             decoder_channels, atrous_rates)
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, [decoder_channels], [num_classes], ['seg'])
        
        
        # Build instance decoder
        instance_decoder_kwargs = dict(
            in_channels=in_channels,
            feature_key=feature_key,
            low_level_channels=low_level_channels,
            low_level_key=low_level_key,
            low_level_channels_project=(64, 32, 16),
            decoder_channels=128,
            atrous_rates=atrous_rates,
            aspp_channels=256
        )
        self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
        
        self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        self.instance_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]
            
        # Instance branch
        instance = self.instance_decoder(features)
        instance = self.instance_head(instance)
        for key in instance.keys():
            pred[key] = instance[key]
                
        return pred

################################################################################################

class BaseSegmentationModel(nn.Module):
    """
    Base class for segmentation models.
    Arguments:
        backbone: A nn.Module of backbone model.
        decoder: A nn.Module of decoder.
    """
    def __init__(self, backbone, decoder, args):
        super(BaseSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.args = args

    def _init_params(self, ):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def set_image_pooling(self, pool_size):
        self.decoder.set_image_pooling(pool_size)

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            result[key] = out
        return result

    def forward(self, x, seg_map=None, label=None, point_list=None, target_shape=None):
        if target_shape is None:
            target_shape = x.shape[-2:]

        # contract: features is a dict of tensors
        features = self.backbone(x)
        pred = self.decoder(features)
        results = self._upsample_predictions(pred, target_shape)
        
        if label is not None: # refined label generation
            
            if self.args.sup == 'point': # point supervision setting
                pseudo_label = refine_label_generation_with_point(
                    results['seg'].clone().detach(), 
                    point_list.cpu().numpy(), 
                    results['offset'].clone().detach(), 
                    label.clone().detach(), 
                    seg_map.clone().detach(),
                    self.args,
                )
                
            else: # image-level supervision setting
                pseudo_label = refine_label_generation(
                    results['seg'].clone().detach(), 
                    results['center'].clone().detach(), 
                    results['offset'].clone().detach(), 
                    label.clone().detach(), 
                    seg_map.clone().detach(),
                    self.args,
                )
            
            return results, pseudo_label
        
        return results


def PanopticDeepLab(backbone, args):
        
    instance_head_kwargs = dict(
            decoder_channels=128,
            head_channels=(128, 32),
            num_classes=(args.num_classes, 2),
            class_key=["center", "offset"], 
        )
    
    decoder = PanopticDeepLabDecoder(in_channels=2048, 
                                     feature_key="res5", 
                                     low_level_channels=(1024, 512, 256), 
                                     low_level_key=["res4", "res3", "res2"], 
                                     low_level_channels_project=(128, 64, 32), 
                                     decoder_channels=256, 
                                     atrous_rates=(3, 6, 9), 
                                     num_classes=args.num_classes+1, 
                                     instance_head_kwargs=instance_head_kwargs
                                    )
    
    model = BaseSegmentationModel(backbone=backbone, decoder=decoder, args=args)
    
    model._init_params()
    
    # set batchnorm momentum
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = args.bn_momentum
    
    return model
        
