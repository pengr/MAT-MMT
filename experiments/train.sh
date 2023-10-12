#!/bin/bash
cd ~/align-mmt
Ckpt_dir=../checkpoints/align-mmt/en-de/ammtl8
Data_save=../data/multi30k/align-mmt/en-de/ammt
Visual_dir=../data/multi30k/features_resnet50

CUDA_VISIBLE_DEVICES=3  python train.py $Data_save \
--task translation_ammt --arch transformer_ammt_tiny --share-all-embeddings --dropout 0.3  \
--optimizer adam --adam-betas 0.9,0.98 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 2000 --lr 0.004 --min-lr 1e-09 --criterion label_smoothed_ammt_cross_entropy --label-smoothing 0.1 \
--loss1-coeff 0.08 --loss2-coeff 0.9 --max-tokens 4096 --save-dir $Ckpt_dir --visual-feature-file $Visual_dir/resnet50-avgpool.npy \
--log-format json --max-update 40000 --find-unused-parameters