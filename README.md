# DCLR
The official implementation of DCLR's code. https://doi.org/10.1016/j.engappai.2026.114901

## Paper: Semi-supervised medical image segmentation method via dual-view graph contrastive learning and latent space uncertainty rectification

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.8 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Train the model
```
python train_Mymodel.py
```

4. Test the model
```
python inference_mydata.py
```


# Acknowledgement

Part of the code is revised from the https://github.com/HiLab-git/SSL4MIS



# Citation:

Dongxu Cheng, Qiwei Dong, Yan Yang, Yan Wang, Ruian Zhu, Yuhui Zheng. Semi-supervised medical image segmentation method via dual-view graph contrastive learning and latent space uncertainty rectification[J]. Engineering Applications of Artificial Intelligence, 2026, 177: 114901.

@article{CHENG2026114901,  
title = {Semi-supervised medical image segmentation method via dual-view graph contrastive learning and latent space uncertainty rectification},  
journal = {Engineering Applications of Artificial Intelligence},   
volume = {177},  
pages = {114901},  
year = {2026},  
issn = {0952-1976},  
author = {Dongxu Cheng and Qiwei Dong and Yan Yang and Yan Wang and Ruian Zhu and Yuhui Zheng},  
}

# Note
The repository is being updated.
