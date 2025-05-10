# A Remote Sensing Change Detection Network Using Visual-Prompt Enhanced CLIP

## Introduction

This repo is the official implementation of  "A Remote Sensing Change Detection Network Using Visual-Prompt Enhanced CLIP"

## Data

Using any change detection dataset you want, but organize dataset path as follows. `dataset_name`  is name of change detection dataset, you can set whatever you want.

```python
dataset_name
├─train
│  ├─label
│  ├─t1
│  └─t2
├─val
│  ├─label
│  ├─t1
│  └─t2
└─test
    ├─label
    ├─t1
    └─t2
```

Below are some binary change detection dataset you may want.

[WHU Building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Paper: Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection

[Weight](https://pan.baidu.com/s/14SJBqFXp-MAOYxHskvFuVg)

Extraction code: gab4 



## Start

For training, run the following code in command line.

`python train.py`

For test and inference, run the following code in command line.

`python inference.py` 

## Config

All the configs of dataset, training, validation and test are put in the file "utils/path_hyperparameter.py", you can change the configs in this file.

# Acknowledgement

​	This repository builds upon the excellent work of offical-SGSLN and BIT. We sincerely thank the authors for their valuable contributions.

## Citation

If you find this work helpful, please consider citing the original repositories that this work builds upon:
@article{zhao2023exchanging, 

title={Exchanging Dual-Encoder--Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization}, 

author={Zhao, Sijie and Zhang, Xueliang and Xiao, Pengfeng and He, Guangjun}, 

journal={IEEE Transactions on Geoscience and Remote Sensing}, 

volume={61}, 

pages={1--16}, 

year={2023},

 publisher={IEEE}

 }



@Article{chen2021a,
    title={Remote Sensing Image Change Detection with Transformers},
    author={Hao Chen, Zipeng Qi and Zhenwei Shi},
    year={2021},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095166}
}

