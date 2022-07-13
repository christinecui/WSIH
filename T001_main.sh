#!/bin/bash
set -e


for i in 16 32 64 128
do

    ##############################
    # COCO
    ##############################
    CUDA_VISIBLE_DEVICES=0 python P000_2_demo.py --nbit $i --dataset 'coco'
    CUDA_VISIBLE_DEVICES=0 python H000_2_demo_att.py --nbit $i --dataset 'coco'
    cd matlab &&
    matlab -nojvm -nodesktop -r "test_save_hash_3($i, 'coco', 'WSIH'); quit;" &&
    cd ..

    ##############################
    # NUS21
    ##############################
    CUDA_VISIBLE_DEVICES=0 python P000_2_demo.py --nbit $i --dataset 'nus21'
    CUDA_VISIBLE_DEVICES=0 python H000_2_demo_att.py --nbit $i --dataset 'nus21'
    cd matlab &&
    matlab -nojvm -nodesktop -r "test_save_hash_3($i, 'nus21','WSIH'); quit;" &&
    cd ..

done

