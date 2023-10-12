#!/bin/bash
cd ~/align-mmt
Data_save=../data/multi30k/align-mmt/en-de/ammt
Data_dir=../data/multi30k/en-de

# extract image labels
#python scripts/extract_image_labels.py --lang en de --trainpref $Data_dir/train.lc.norm.tok.bpe \
#--validpref $Data_dir/val.lc.norm.tok.bpe --testpref $Data_dir/test_2016_flickr.lc.norm.tok.bpe,\
#$Data_dir/test_2017_flickr.lc.norm.tok.bpe,$Data_dir/test_2017_mscoco.lc.norm.tok.bpe \
#--batch-size 2048 --no-reconst-orig

# create image indexs
#python scripts/create_image_indexs.py --src en --trainpref $Data_dir/train.lc.norm.tok.bpe \
#--validpref $Data_dir/val.lc.norm.tok.bpe --testpref $Data_dir/test_2016_flickr.lc.norm.tok.bpe,\
#$Data_dir/test_2017_flickr.lc.norm.tok.bpe,$Data_dir/test_2017_mscoco.lc.norm.tok.bpe

# with bpe (en-de)
python preprocess_ammt.py --source-lang en --target-lang de --trainpref $Data_dir/train.lc.norm.tok.bpe \
--validpref $Data_dir/val.lc.norm.tok.bpe --testpref $Data_dir/test_2016_flickr.lc.norm.tok.bpe,\
$Data_dir/test_2017_flickr.lc.norm.tok.bpe,$Data_dir/test_2017_mscoco.lc.norm.tok.bpe \
--destdir $Data_save --label-suffix label --image-suffix image --joined-dictionary --workers 16