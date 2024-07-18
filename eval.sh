#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --csv_path_test dataset/ARKitFace/ARKitFace_list/list/ARKitFace_test.csv \
	--num_thread 8 \
	--batch_size 64 \
	--dataset_mode biwi \
	--epoch 30 \
	--name trg_240717