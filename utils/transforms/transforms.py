import random
import warnings
import cv2
import math
import numpy as np
from torchvision.transforms import functional as F
import torch

class Compose(object):
    """
    Composes a sequence of transforms.
    Arguments:
        transforms: A list of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, seg_map, peak):
        for t in self.transforms:
            image, seg_map, peak = t(image, seg_map, peak)
        return image, seg_map, peak

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    """
    Converts image to torch Tensor.
    """
    def __call__(self, image, seg_map, peak):
        
        image = F.to_tensor(image)
        
        seg_map = np.array(seg_map, dtype=np.uint8, copy=True)

        seg_map = torch.from_numpy(seg_map)
        
        return image, seg_map, peak
        
        
class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, seg_map, peak):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        image = F.normalize(image, self.mean, self.std)
        
        peak = [ [int(x), int(y), cls, conf] for x, y, cls, conf in peak]
        
        return image, seg_map, peak


class RandomScale(object):
    """
    Applies random scale augmentation.
    Arguments:
        min_scale: Minimum scale value.
        max_scale: Maximum scale value.
        scale_step_size: The step size from minimum to maximum value.
    """
    def __init__(self, min_scale, max_scale, scale_step_size):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

    @staticmethod
    def get_random_scale(min_scale_factor, max_scale_factor, step_size):
        """Gets a random scale value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        Raises:
            ValueError: min_scale_factor has unexpected value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')

        if min_scale_factor == max_scale_factor:
            return min_scale_factor

        # When step_size = 0, we sample the value uniformly from [min, max).
        if step_size == 0:
            return random.uniform(min_scale_factor, max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
        np.random.shuffle(scale_factors)
        return scale_factors[0]

    #def __call__(self, image, label):
    def __call__(self, image, seg_map, peak):
        f_scale = self.get_random_scale(self.min_scale, self.max_scale, self.scale_step_size)
        # TODO: cv2 uses align_corner=False
        # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)
        image_dtype = image.dtype
        seg_map_dtype = seg_map.dtype
        
        image = cv2.resize(image.astype(np.float), None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        
        seg_map = cv2.resize(seg_map.astype(np.float), None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        
        peak = [ [p[0]*f_scale, p[1]*f_scale, p[2], p[3]] for p in peak ]
        
        return image.astype(image_dtype), seg_map.astype(seg_map_dtype), peak


class RandomCrop(object):
    """
    Applies random crop augmentation.
    Arguments:
        crop_h: Integer, crop height size.
        crop_w: Integer, crop width size.
        pad_value: Tuple, pad value for image, length 3.
        ignore_label: Tuple, pad value for label, length could be 1 (semantic) or 3 (panoptic).
        random_pad: Bool, when crop size larger than image size, whether to randomly pad four boundaries,
            or put image to top-left and only pad bottom and right boundaries.
    """
    def __init__(self, crop_h, crop_w, pad_value, ignore_label, random_pad):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_label = ignore_label
        self.random_pad = random_pad

    def __call__(self, image, seg_map, peak):
        img_h, img_w = image.shape[0], image.shape[1]
        # save dtype
        image_dtype = image.dtype
        seg_map_dtype = seg_map.dtype

        # padding
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
                
            img_pad = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                         value=self.pad_value)
            seg_map_pad = cv2.copyMakeBorder(seg_map, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
            
            peak = [ [px+pad_left, py+pad_top, cls, conf] for px, py, cls, conf in peak ]
            
        else:
            img_pad, seg_map_pad = image, seg_map
            
        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        
        image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        seg_map = np.asarray(seg_map_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        
        peak_crop = []
        for px, py, cls, conf in peak:
            if (h_off <= py < h_off + self.crop_h) and (w_off <= px < w_off + self.crop_w):
                px_crop, py_crop = px-w_off, py-h_off
                
                #if (0 <= px_crop < self.crop_w) and (0 <= py_crop < self.crop_h):
                peak_crop.append( [px_crop, py_crop, cls, conf]  )
        
        return image.astype(image_dtype), seg_map.astype(seg_map_dtype), peak_crop


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, image, seg_map, peak):

        if random.random() < 0.5:
            image = torch.flip(image, [2])
            seg_map = torch.flip(seg_map, [1])
            
            _, H, W = image.shape
            
            peak = [ [W-p[0]-1, p[1], p[2], p[3]] for p in peak ]
            
        return image, seg_map, peak

    
class Resize(object):
    """
    Applies random scale augmentation.
    Reference: https://github.com/tensorflow/models/blob/master/research/deeplab/input_preprocess.py#L28
    Arguments:
        min_resize_value: Desired size of the smaller image side, no resize if set to None
        max_resize_value: Maximum allowed size of the larger image side, no limit if set to None
        resize_factor: Resized dimensions are multiple of factor plus one.
        keep_aspect_ratio: Boolean, keep aspect ratio or not. If True, the input
            will be resized while keeping the original aspect ratio. If False, the
            input will be resized to [max_resize_value, max_resize_value] without
            keeping the original aspect ratio.
        align_corners: If True, exactly align all 4 corners of input and output.
    """
    def __init__(self, min_resize_value=None, max_resize_value=None, resize_factor=None,
                 keep_aspect_ratio=True, align_corners=False):
        if min_resize_value is not None and min_resize_value < 0:
            min_resize_value = None
        if max_resize_value is not None and max_resize_value < 0:
            max_resize_value = None
        if resize_factor is not None and resize_factor < 0:
            resize_factor = None
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.keep_aspect_ratio = keep_aspect_ratio
        self.align_corners = align_corners

        if self.align_corners:
            warnings.warn('`align_corners = True` is not supported by opencv.')

        if self.max_resize_value is not None:
            # Modify the max_size to be a multiple of factor plus 1 and make sure the max dimension after resizing
            # is no larger than max_size.
            if self.resize_factor is not None:
                self.max_resize_value = (self.max_resize_value - (self.max_resize_value - 1) % self.resize_factor)

    def __call__(self, image, seg_map, peak):
        if self.min_resize_value is None:
            return image, label
        [orig_height, orig_width, _] = image.shape
        orig_min_size = np.minimum(orig_height, orig_width)

        # Calculate the larger of the possible sizes
        large_scale_factor = self.min_resize_value / orig_min_size
        large_height = int(math.floor(orig_height * large_scale_factor))
        large_width = int(math.floor(orig_width * large_scale_factor))
        large_size = np.array([large_height, large_width])

        new_size = large_size
        if self.max_resize_value is not None:
            # Calculate the smaller of the possible sizes, use that if the larger is too big.
            orig_max_size = np.maximum(orig_height, orig_width)
            small_scale_factor = self.max_resize_value / orig_max_size
            small_height = int(math.floor(orig_height * small_scale_factor))
            small_width = int(math.floor(orig_width * small_scale_factor))
            small_size = np.array([small_height, small_width])

            if np.max(large_size) > self.max_resize_value:
                new_size = small_size

        # Ensure that both output sides are multiples of factor plus one.
        if self.resize_factor is not None:
            new_size += (self.resize_factor - (new_size - 1) % self.resize_factor) % self.resize_factor
            # If new_size exceeds largest allowed size
            new_size[new_size > self.max_resize_value] -= self.resize_factor

        if not self.keep_aspect_ratio:
            # If not keep the aspect ratio, we resize everything to max_size, allowing
            # us to do pre-processing without extra padding.
            new_size = [np.max(new_size), np.max(new_size)]

        # TODO: cv2 uses align_corner=False
        # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)
        image_dtype = image.dtype
        seg_map_dtype = seg_map.dtype
        
        image = cv2.resize(image.astype(np.float), (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)
        seg_map = cv2.resize(seg_map.astype(np.float), (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        
        peak = [ [int(p[0]/orig_width*new_size[1]), int(p[1]/orig_height*new_size[0]), p[2], p[3]] for p in peak ]
        
        return image.astype(image_dtype), seg_map.astype(seg_map_dtype), peak


    
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, seg_map, peak):
        if random.random() < 0.5:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image[image>255] = 255
            image[image<0] = 0
            
        return image, seg_map, peak
    
class RandomBrightness(object):
    def __init__(self, delta=16):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, seg_map, peak):
        if random.random() < 0.5:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image>255] = 255
            image[image<0] = 0
            
        return image, seg_map, peak

class RandomHue(object):
    def __init__(self, delta=36.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, seg_map, peak):
        if random.random() < 0.5:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, seg_map, peak


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, seg_map, peak):
        if random.random() < 0.5:
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, seg_map, peak


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, seg_map, peak):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, seg_map, peak

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current='RGB', transform='HSV'),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, seg_map, peak):
        #m = image.copy()
        im = image.copy().astype(np.float32)
        
        im, seg_map, peak = self.rand_brightness(im, seg_map, peak)
        
        if random.random() < 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
            
        im, seg_map, peak = distort(im, seg_map, peak)
        # im, boxes, labels = self.rand_light_noise(im, boxes, labels)
        
        im = np.clip(im, 0, 255).astype(np.uint8)
        
        return im, seg_map, peak

    
class Keepsize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, shape=None):
        self.shape = shape
        
        pass

    def __call__(self, img, seg_map, peak):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        ori_h, ori_w, _ = img.shape

        if self.shape is None:
            new_h = (ori_h + 31) // 32 * 32
            new_w = (ori_w + 31) // 32 * 32
        else:
            new_h = self.shape[1]
            new_w = self.shape[0]
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        seg_map = cv2.resize(seg_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        peak = [ [int(p[0]/ori_w*new_w), int(p[1]/ori_h*new_h), p[2], p[3]] for p in peak ]
        
        return img, seg_map, peak
