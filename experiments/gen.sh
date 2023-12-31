#!/bin/bash
cd ~/align-mmt
Ckpt_dir=../checkpoints/align-mmt/en-de/ammtl8
Data_save=../data/multi30k/align-mmt/en-de/ammt
Data_dir=../data/multi30k/en-de
Meteor_dir=../meteor-1.5/meteor-1.5.jar

#if [ ! -f "$Ckpt_dir/checkpoint_last10_avg.pt" ]; then
#  python scripts/average_checkpoints.py \
#  --inputs $Ckpt_dir \
#  --output $Ckpt_dir/checkpoint_last10_avg.pt \
#  --num-epoch-checkpoints  10 \
#  --checkpoint-upper-bound 290
#fi

for f in test test1 test2; do
    CUDA_VISIBLE_DEVICES=1 python -W ignore generate.py $Data_save \
    --task translation_ammt --criterion label_smoothed_ammt_cross_entropy \
    --path $Ckpt_dir/checkpoint_last10_avg.pt --gen-subset $f --beam 5 --batch-size 128 \
    --remove-bpe --results-path $Ckpt_dir
done

for f in valid test test1 test2; do
    rm -f $Ckpt_dir/$f.txt
    grep ^H $Ckpt_dir/generate-$f.txt | sort -n -k 2 -t '-' | cut -f 3 >> $Ckpt_dir/$f.txt
    cat $Ckpt_dir/generate-$f.txt | tail -1

    if [ $f = valid ]; then
        java -Xmx2G -jar $Meteor_dir $Ckpt_dir/$f.txt $Data_dir/val.lc.norm.tok.de -l de -norm -q -vOut | tail -1
    elif [ $f = test ]; then
        java -Xmx2G -jar $Meteor_dir $Ckpt_dir/$f.txt $Data_dir/test_2016_flickr.lc.norm.tok.de -l de -norm -q -vOut | tail -1
    elif [ $f = test1 ]; then
        java -Xmx2G -jar $Meteor_dir $Ckpt_dir/$f.txt $Data_dir/test_2017_flickr.lc.norm.tok.de -l de -norm -q -vOut | tail -1
    else
        java -Xmx2G -jar $Meteor_dir $Ckpt_dir/$f.txt $Data_dir/test_2017_mscoco.lc.norm.tok.de -l de -norm -q -vOut | tail -1
    fi
done