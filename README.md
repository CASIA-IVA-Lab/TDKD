
# TDKD
---------------------------------------

Task Decoupled Knowledge Distillation For Lightweight Face Detectors (Accepted by ACM MM2020)


## Abstract
We propose a knowledge distillation method for the face detection task. This method decouples the distillation task of face detection into two subtasks, i.e., the classification distillation subtask and the regression distillation subtask. We add the task-specific convolutions in the teacher network and add the adaption convolutions on the feature maps of the student network to generate the task decoupled features. Then, each subtask uses different samples in distilling the features to be consistent with the corresponding detection subtask. Moreover, we propose an effective probability distillation method to joint boost the accuracy of the student network.

## WiderFace Val Performance in single scale.
| Model | easy | medium | hard |
|:-|:-:|:-:|:-:|
| RetinaFace-Mobilenet0.25 (Student) | 87.1% | 85.7% | 79.2% |
| RetinaFace-Mobilenet0.25 (TDKD) | 88.9% | 87.5% | 81.5% |



## Installation
1. To folder where you want to download this repo
```shell
cd /your/own/path/
```

2. Run
```Shell
git clone https://github.com/CASIA-IVA-Lab/TDKD.git
```

3. Install dependencies:
    - [pytorch>=1.1.0](https://pytorch.org/)
    - torchvision>=0.3.0+
    - Python 3


## Data
1. Download the [WIDER FACE](https://pan.baidu.com/s/1BB86wsXx_2B8eLbC8RrMPA) dataset. [password: qisa]
2. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```

## Train
1. In the code, we integrate the teacher model and the pre-trained model. So just run the following command to start distillation:
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python train.py
  ```

## Evaluation
1. Generate txt file
```Shell
python test_widerface.py
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```



## Reference

- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
```
@inproceedings{LiangZZJTW20,
  title = {Task Decoupled Knowledge Distillation For Lightweight Face Detectors},
  author = {Liang, Xiaoqing and Zhao, Xu and Zhao, Chaoyang and Jiang, Nanfei and Tang, Ming and Wang, Jinqiao},
  booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia, Virtual Event / Seattle, WA, USA, October 12-16, 2020},
  year = {2020}
}
```
