# Peak Attention Module (PAM)

## Abtract

CAMs have a limitation in obtaining the accurate instance cue because several instance cues might be extracted in a single instance due to noisy activation regions as illustrated in the bellow Figure. 
It disturbs the generation of pseudo instance labels.
To address this limitation, we propose a peak attention module (PAM) to extract one appropriate instance cue per instance. 
PAM aims to strengthen the attention on peak regions, while weakening the attention on noisy activation regions.

<img src = "https://github.com/clovaai/BESTIE/blob/main/figures/PAM_architecture.png" width="50%" height="50%">

<img src = "https://github.com/clovaai/BESTIE/blob/main/figures/PAM_comparison.png" width="50%" height="50%">


## How to Run?

```
# change the data ROOT in the shell script
bash run_PAM.sh
```

Note that extracted peak points are used in the image-level supervised BESTIE.
We provide the weight for the pretrained classfier with PAM module [[download]](https://drive.google.com/file/d/1I5DocPV2Lkc59DtDrr4XoQuVlKdRi4km/view?usp=sharing)

## Acknowledgement

Our implementation is based on these repositories:
- (DRS) https://github.com/qjadud1994/DRS