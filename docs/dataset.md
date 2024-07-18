# Datasets

## ARKitFace

You can download the ARKitFace dataset from the following link: [ARKitFace dataset](https://github.com/cbsropenproject/6dof_face).
Follow the instructions in the `Download Dataset` section on the linked page. 
You will be able to obtain both the dataset and the preprocessed files. 
We used the files downloaded from ARKitFace without any modifications.

## 300W-LP

You can download the 300W-LP dataset from the following link: [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
We used the `create_filename_list.py` script from [TokenHPE](https://github.com/zc2023/TokenHPE) to preprocess the 300W-LP dataset. Running `create_filename_list.py` will generate a `files.txt` file. 
For convenience, you can download the `files.txt` from [here](https://drive.google.com/file/d/1ApNOj7lvk-c4yGtIL4LiEcgEbn0oqk-g/view?usp=drive_link).

Place the downloaded file under the 300W_LP directory as follows: `dataset/300W_LP/files.txt`.

## BIWI

You can download the BIWI dataset from the following link: [BIWI](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html).
To run our code, you need the file for the predicted bounding box. You can download this file from [here](https://drive.google.com/file/d/1u2lE0unRyz9HlPZC6SczwVaF89iwj2iZ/view?usp=drive_link).

Place the downloaded bounding box file in the following location: `dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb/annot_mtcnn_fan.pkl`.