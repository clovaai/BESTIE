# Training BESTIE with image-level labels.

ROOT=your_dataset_root_path
SUP=cls
PSEUDO_THRESH=0.7
REFINE_THRESH=0.3
REFINE_WARMUP=0
SIZE=416
BATCH=16
WORKERS=4
TRAIN_ITERS=50000
BACKBONE=hrnet48 # [resnet50, resnet101, hrnet32, hrnet48]
VAL_IGNORE=False

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--root_dir ${ROOT} --sup ${SUP} --batch_size ${BATCH} --num_workers ${WORKERS} --crop_size ${SIZE} --train_iter ${TRAIN_ITERS} \
--refine True --refine_iter ${REFINE_WARMUP} --pseudo_thresh ${PSEUDO_THRESH} --refine_thresh ${REFINE_THRESH} \
--val_freq 1000 --val_ignore ${VAL_IGNORE} --val_clean False --val_flip False \
--seg_weight 1.0 --center_weight 200.0 --offset_weight 0.01 \
--lr 5e-5 --backbone ${BACKBONE} --random_seed 3407

