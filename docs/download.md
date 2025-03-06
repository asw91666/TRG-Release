# Datasets

## ARKitFace

You can download the ARKitFace dataset from the following link: [ARKitFace dataset](https://github.com/cbsropenproject/6dof_face).
Follow the instructions in the `Download Dataset` section on the linked page. 
You will be able to obtain both the dataset and the preprocessed files. 
We used the files downloaded from ARKitFace without any modifications.

To run our code, you need the flip index file for training. You can download this file from [here](https://drive.google.com/drive/folders/1Qdj62asVBpj-cGM2mnymD1bqNEBhf6RZ?usp=sharing).

Place the file in the following location: `dataset/ARKitFace/flip_index.npy`.

```
${ROOT}  
|-- dataset  
|   |-- 300W_LP
|   |-- BIWI
|   |-- ARKitFace
|   |   |-- ARKitFace_info_test
|   |   |-- ARKitFace_info_train
|   |   |-- ARKitFace_list
|   |   |-- ARKitFace_resources
|   |   |-- ARKitFace_image_test
|   |   |-- ARKitFace_image_train
|   |   |-- flip_index.npy
|-- ... 
|-- ... 
```

## 300W-LP

You can download the 300W-LP dataset from the following link: [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
We used the `create_filename_list.py` script from [TokenHPE](https://github.com/zc2023/TokenHPE) to preprocess the 300W-LP dataset. Running `create_filename_list.py` will generate a `files.txt` file. 
For convenience, you can download the `files.txt` from [here](https://drive.google.com/drive/folders/1Qdj62asVBpj-cGM2mnymD1bqNEBhf6RZ?usp=sharing).

Place the downloaded file under the 300W_LP directory as follows: `dataset/300W_LP/files.txt`.

```
${ROOT}  
|-- dataset  
|   |-- BIWI
|   |-- ARKitFace
|   |-- 300W_LP
|   |   |-- IBUG
|   |   |-- AFW
|   |   |-- LFPW
|   |   |-- IBUG_Flip
|   |   |-- AFW_Flip
|   |   |-- HELEN
|   |   |-- LFPW_Flip
|   |   |-- HELEN_Flip
|   |   |-- landmarks
|   |   |   |-- AFW
|   |   |   |-- HELEN
|   |   |   |-- IBUG
|   |   |   |-- LFPW
|   |   |-- files.txt
|-- ... 
|-- ... 
```

## BIWI

You can download the BIWI dataset from the following link: [BIWI](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database).
To run our code, you need the file for the predicted bounding box. You can download this file from [here](https://drive.google.com/drive/folders/1Qdj62asVBpj-cGM2mnymD1bqNEBhf6RZ?usp=sharing).

Place the downloaded bounding box file in the following location: `dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb/annot_mtcnn_fan.pkl`.

```
${ROOT}  
|-- dataset  
|   |-- ARKitFace
|   |-- 300W_LP
|   |-- BIWI
|   |   |-- download_from_official_site
|   |   |   |-- db_annotations
|   |   |   |-- head_pose_masks
|   |   |   |-- kinect_head_pose_db
|   |   |   |   |-- hpdb
|   |   |   |   |   |-- annot_mtcnn_fan.pkl
|   |   |   |   |   |-- 01
|   |   |   |   |   |-- 02
|   |   |   |   |   |-- 03
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- ...
|-- ... 
|-- ... 

```

# Pretrained weights
| Model                   | Train Dataset     | Link                |
| ----------------------- | ----------------- | ------------------- |
| TRG                     | ARKitFace         | [Download](https://drive.google.com/drive/folders/1CQeB4W2KbNVpzx4FZfuyMVgljlOvtpwU?usp=sharing)        |
| TRG*                    | ARKitFace+300WLP  | [Download](https://drive.google.com/drive/folders/1eymumNT307y7nyvMsZD0wuSPtmEkYyWv?usp=sharing)        |

```
${ROOT}  
|-- checkpoint
|   |-- trg_240717
|   |   |-- checkpoint-30/
|   |-- trg_single_240717
|   |   |-- checkpoint-30/
|-- ... 
|-- ... 
```
