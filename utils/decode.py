import numpy as np
import torch

def colorize_offset(offset_map, offset_weight, seg_map=None, pred=True):

    import matplotlib.colors
    import math

    a = (np.arctan2(-offset_map[0], -offset_map[1]) / math.pi + 1) / 2

    r = np.sqrt(offset_map[0] ** 2 + offset_map[1] ** 2)
    s = r / (np.max(r) + 1e-5)
    
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1)
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)
    rgb_color = np.uint8(rgb_color * 255)
    
    if seg_map is not None:
        rgb_color[np.where(seg_map == 0)] = [0, 0, 0] # background
    
    if not pred:
        rgb_color[np.where(offset_weight == 0)] = [255, 255, 255] # ignore
    
    return rgb_color


def voc_names():
    return [
        "background", "aeroplane", "bicycle", "bird",
        "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]


def get_palette():
    palette = []
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21] = np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8').flatten()
    
    return palette

def voc_colors():
    colors = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128], 
                       [255, 255, 255],
                      [200, 200, 200]], dtype='uint8')
    return colors


def cam_to_seg(CAM, sal_map, palette, alpha=0.2, ignore=255):
    colors = voc_colors()
    C, H, W = CAM.shape
    
    CAM[CAM < alpha] = 0 # object cue

    bg = np.zeros((1, H, W), dtype=np.float32)
    pred_map = np.concatenate([bg, CAM], axis=0)  # [21, H, W]
    
    pred_map[0, :, :] = (1. - sal_map) # backgroudn cue
    
    # conflict pixels with multiple confidence values
    bg = np.array(pred_map > 0.99, dtype=np.uint8)
    bg = np.sum(bg, axis=0)
    pred_map = pred_map.argmax(0).astype(np.uint8)
    pred_map[bg > 2] = ignore

    # pixels regarded as background but confidence saliency values 
    bg = (sal_map == 1).astype(np.uint8) * (pred_map == 0).astype(np.uint8) # and operator
    pred_map[bg > 0] = ignore
    
    pred_map = np.uint8(pred_map)
    
    palette = get_palette()
    pred_map = Image.fromarray(pred_map)
    pred_map.putpalette(palette)
                
    return pred_map


def cam_with_crf(cam, img, keys, fg_thresh=0.1, bg_thresh=0.001):
    cam = np.float32(cam) / 255.

    cam = cam[keys] # valid category selection
    
    valid_cat = np.pad(keys + 1, (1, 0), mode='constant') # valid category : [background, val_cat+1]

    fg_conf_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=fg_thresh) # [c+1, H, W]
    fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
    pred = crf_inference_label(img, fg_conf_cam, n_labels=valid_cat.shape[0])
    fg_conf = valid_cat[pred] # convert to whole index (0, 1, 2) -> (0 ~ 20)

    bg_conf_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=bg_thresh) # [c+1, H, W]
    bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
    pred = crf_inference_label(img, bg_conf_cam, n_labels=valid_cat.shape[0])
    bg_conf = valid_cat[pred] # convert to whole index (0, 1, 2) -> (0 ~ 20)

    conf = fg_conf.copy()
    conf[fg_conf == 0] = 21
    conf[bg_conf + fg_conf == 0] = 0 # both zero

    conf_color = colors[conf]
    
    return conf, conf_color


def heatmap_colorize(score_map, exclude_zero=True, normalize=True):
    import matplotlib.colors
    
    VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                 (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                 (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

    if exclude_zero:
        VOC_color = VOC_color[1:]

    test = VOC_color[np.argmax(score_map, axis=0)%22]
    test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test
    if normalize:
        test /= np.max(test) + 1e-5

    return test

def decode_seg_map_sequence(label_masks):
    if label_masks.ndim == 2:
        label_masks = label_masks[None, :, :]
    
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    
    n_classes = 21
    label_colours = get_pascal_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_ins_colors():
    ins_colors = np.random.random((2000, 3))
    ins_colors = np.uint8(ins_colors*255)
    ins_colors[0] = [0, 0, 0]
    ins_colors[1] = [192, 128, 0]
    ins_colors[2] = [64, 0, 128]
    ins_colors[3] = [192, 0, 128]
    ins_colors[4] = [64, 128, 128]
    ins_colors[5] = [192, 128, 128]
    ins_colors[6] = [0, 64, 0]
    ins_colors[7] = [128, 64, 0]
    ins_colors[8] = [0, 192, 0]
    ins_colors[9] = [128, 192, 0]
    ins_colors[10] = [0, 64, 128]
    ins_colors[11] = [128, 0, 0]
    ins_colors[12] = [0, 128, 0]
    ins_colors[13] = [128, 128, 0]
    ins_colors[14] = [0, 0, 128]
    ins_colors[15] = [128, 0, 128]
    ins_colors[16] = [0, 128, 128]
    ins_colors[17] = [128, 128, 128]
    ins_colors[18] = [64, 0, 0]
    ins_colors[19] = [192, 0, 0]
    ins_colors[20] = [64, 128, 0]
    return ins_colors