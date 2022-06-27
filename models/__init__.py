from .panoptic_deeplab import PanopticDeepLab
from .hrnet import hrnet32, hrnet48
from .resnet import resnet50, resnet101

def model_factory(args):
    
    if args.backbone == 'hrnet32':
        model = hrnet32(args)
    elif args.backbone == 'hrnet48':
        model = hrnet48(args)

    elif args.backbone == 'resnet50':
        backbone = resnet50(pretrained=True)
        model = PanopticDeepLab(backbone, args)
        
    elif args.backbone == 'resnet101':
        backbone = resnet101(pretrained=True)
        model = PanopticDeepLab(backbone, args)
        
    return model
        