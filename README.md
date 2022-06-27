# BESTIE - Official Pytorch Implementation (CVPR 2022)

**Beyond Semantic to Instance Segmentation: Weakly-Supervised Instance Segmentation via Semantic Knowledge Transfer and Self-Refinement (CVPR 2022)** <br />
Beomyoung Kim<sup>1</sup>, YoungJoon Yoo<sup>1,2</sup>, Chaeeun Rhee<sup>3</sup>, Junmo Kim<sup>4</sup><br>

<sup>1</sup> <sub>NAVER CLOVA</sub><br />
<sup>2</sup> <sub>NAVER AI Lab</sub><br />
<sup>3</sup> <sub>Inha University</sub><br />
<sup>4</sup> <sub>KAIST</sub><br />

[![Paper](https://img.shields.io/badge/arXiv-2109.09477-brightgreen)](https://arxiv.org/abs/2109.09477)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-semantic-to-instance-segmentation/image-level-supervised-instance-segmentation)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation?p=beyond-semantic-to-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-semantic-to-instance-segmentation/image-level-supervised-instance-segmentation-2)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation-2?p=beyond-semantic-to-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-semantic-to-instance-segmentation/image-level-supervised-instance-segmentation-1)](https://paperswithcode.com/sota/image-level-supervised-instance-segmentation-1?p=beyond-semantic-to-instance-segmentation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-semantic-to-instance-segmentation/point-supervised-instance-segmentation-on)](https://paperswithcode.com/sota/point-supervised-instance-segmentation-on?p=beyond-semantic-to-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-semantic-to-instance-segmentation/point-supervised-instance-segmentation-on-2)](https://paperswithcode.com/sota/point-supervised-instance-segmentation-on-2?p=beyond-semantic-to-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-semantic-to-instance-segmentation/point-supervised-instance-segmentation-on-1)](https://paperswithcode.com/sota/point-supervised-instance-segmentation-on-1?p=beyond-semantic-to-instance-segmentation)

<img src = "https://github.com/clovaai/BESTIE/blob/main/figures/overview.png" width="100%" height="100%">


# Abtract

Weakly-supervised instance segmentation (WSIS) has been considered as a more challenging task than weakly-supervised semantic segmentation (WSSS). Compared to WSSS, WSIS requires instance-wise localization, which is difficult to extract from image-level labels. To tackle the problem, most WSIS approaches use off-the-shelf proposal techniques that require pre-training with instance or object level labels, deviating the fundamental definition of the fully-image-level supervised setting. 
In this paper, we propose a novel approach including two innovative components. First, we propose a *semantic knowledge transfer* to obtain pseudo instance labels by transferring the knowledge of WSSS to WSIS while eliminating the need for the off-the-shelf proposals. Second, we propose a *self-refinement* method to refine the pseudo instance labels in a self-supervised scheme and to use the refined labels for training in an online manner. Here, we discover an erroneous phenomenon, *semantic drift*, that occurred by the missing instances in pseudo instance labels categorized as background class. This *semantic drift* occurs confusion between background and instance in training and consequently degrades the segmentation performance. We term this problem as *semantic drift problem* and show that our proposed *self-refinement* method eliminates the semantic drift problem.
The extensive experiments on PASCAL VOC 2012 and MS COCO demonstrate the effectiveness of our approach, and we achieve a considerable performance without off-the-shelf proposal techniques. The code is available at https://github.com/clovaai/BESTIE.

# Experimental Results (VOC 2012, COCO)

<img src = "https://github.com/clovaai/BESTIE/blob/main/figures/result_voc.png" width="50%" height="50%">
<img src = "https://github.com/clovaai/BESTIE/blob/main/figures/result_coco.png" width="50%" height="50%">

* BESTIE (HRNet48, Image-label) : 42.6 mAP50 on VOC2012 [[download]](https://drive.google.com/drive/folders/1T3Cy3ybQ1Vk1Kow4QRm1lNRuTnoIhBBH?usp=sharing)
* BESTIE (HRNet48, point-label) : 46.7 mAP50 on VOC2012 [[download]](https://drive.google.com/drive/folders/1T3Cy3ybQ1Vk1Kow4QRm1lNRuTnoIhBBH?usp=sharing)


# Qualitative Results

<img src = "https://github.com/clovaai/BESTIE/blob/main/figures/qualitavie_result.png" width="100%" height="100%">


# News

- [x] official pytorch code release
- [ ] release the code for the classifier with PAM module
- [ ] update training code and dataset for COCO

# How To Run

### Requirements
- torch>=1.10.1
- torchvision>=0.11.2
- chainercv>=0.13.1
- numpy
- pillow
- scikit-learn
- tqdm

### Datasets

- Download Pascal VOC2012 dataset from the [official dataset homepage](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
- Download other data from [[here]](https://drive.google.com/drive/folders/1NCzMAZzxvDu8E6qV1uDIrts9ZuHpmGgV?usp=sharing).
  - `Center_points/` (ground-trugh point labels)
  - `Peak_points/` (point labels extracted by PAM module and image-level labels)
  - `WSSS_maps/` (weakly-supervised semantic segmentation outputs)

```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationObject/
        --- Center_points/
        --- Peak_points/
        --- WSSS_maps/
```

### Image-level Supervised Instance Segmentation on VOC2012
```
# change the data ROOT in the shell script
bash scrips/run_image_labels.sh
```

### Point Supervised Instance Segmentation on VOC2012
```
# change the data ROOT in the shell script
bash scrips/run_point_labels.sh
```

### Mask R-CNN Refinement

1. Generate COCO-style pseudo labels using the BESTIE model.
2. Train the Mask R-CNN using the pseudo-labels: https://github.com/facebookresearch/maskrcnn-benchmark .


# Acknowledgement

Our implementation is based on these repositories:
- (Panoptic-DeepLab) https://github.com/bowenc0221/panoptic-deeplab
- (HRNet) https://github.com/HRNet/HRNet-Human-Pose-Estimation
- (DRS) https://github.com/qjadud1994/DRS

# License

```
Copyright (c) 2022-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
