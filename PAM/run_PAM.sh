CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --root_dir=your_dataset_root_path \
    --lr=0.001 \
    --epoch=15 \
    --decay_points='5,10' \
    --alpha=0.7 \
    --save_folder=checkpoints/PAM \
    --show_interval=50

CUDA_VISIBLE_DEVICES=0 python point_extraction.py \
    --root_dir=your_dataset_root_path \
    --alpha=0.7 \
    --checkpoint=checkpoints/PAM/ckpt_15.pth \
    --save_dir Peak_points
    