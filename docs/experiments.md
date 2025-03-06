# Train

If you want to train the TRG model on the ARKitFace dataset, use the script below:
```bash
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--gpu_ids 0 \
	--display_freq 100 \
	--name trg \
	--csv_path_train dataset/ARKitFace/ARKitFace_list/list/ARKitFace_train.csv \
	--csv_path_test dataset/ARKitFace/ARKitFace_list/list/ARKitFace_test.csv \
	--num_threads 16 \
	--batch_size 128 \
	--n_epochs 20 \
	--n_epochs_decay 10 \
	--dataset_mode arkit \
	--img_size 192 \
	--lr 1e-4 \
  --loop_loss_weight 0.5 \
  --vertex_world_weight 20.0 \
	--vertex_cam_weight 2.0 \
  --vertex_img_weight 0.01 \
	--lmk68_loss_weight 1.25 \
	--rotmat_loss_weight 10.0 \
	--edge_weight 2.0 \
	--lmk5_2d_loss_weight 0.1
```

If you want to train the TRG model on the ARKitFace+300WLP dataset, use the script below:
```bash
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--gpu_ids 0 \
	--display_freq 100 \
	--name trg_multi \
	--csv_path_train dataset/ARKitFace/ARKitFace_list/list/ARKitFace_train.csv \
	--csv_path_test dataset/ARKitFace/ARKitFace_list/list/ARKitFace_test.csv \
	--num_threads 16 \
	--batch_size 128 \
	--n_epochs 20 \
	--n_epochs_decay 10 \
	--dataset_mode multiple \
	--img_size 192 \
	--lr 1e-4 \
  --loop_loss_weight 0.5 \
  --vertex_world_weight 20.0 \
	--vertex_cam_weight 2.0 \
  --vertex_img_weight 0.01 \
	--lmk68_loss_weight 1.25 \
	--rotmat_loss_weight 10.0 \
	--edge_weight 2.0 \
	--lmk5_2d_loss_weight 0.1
```

# Test

You can evaluate the TRG model which is trained on ARKitFace dataset on ARKitFace test dataset.

```bash
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --csv_path_test dataset/ARKitFace/ARKitFace_list/list/ARKitFace_test.csv \
	--num_thread 8 \
	--batch_size 64 \
	--dataset_mode arkit \
	--epoch 30 \
	--name trg_single_240717
```

You can evaluate the TRG model which is trained on ARKitFace dataset on BIWI dataset.

```bash
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --csv_path_test dataset/ARKitFace/ARKitFace_list/list/ARKitFace_test.csv \
	--num_thread 8 \
	--batch_size 64 \
	--dataset_mode biwi \
	--epoch 30 \
	--name trg_single_240717
```