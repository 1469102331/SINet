#  Spatial Invertible Network With Mamba-Convolution for  Hyperspectral Image Fusion

 The code for 《 Spatial Invertible Network With Mamba-Convolution for  Hyperspectral Image Fusion》

![Language](https://img.shields.io/badge/language-python-brightgreen) 

Our model was trained on an NVIDIA A800-SXM4-80GB GPU.

<div align="center">
    <img src="SINet.png" alt="framework" width="800"/>
</div>

## 👉 Data
  NOTE:

* [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/)

* [Harvard](http://vision.seas.harvard.edu/hyperspec/)

* [Pavia](https://github.com/liangjiandeng/HyperPanCollection)



## 🌿 Getting Started

### Environment Setup

To get started, we recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment.
    
    conda create -n sinet_env python==3.11
    
    conda activate sinet_env 
    
    pip install -r requirements.txt
    
    

### Train 
    
    python Train_cave.py

### Test
    python Test.py

### Citation
If this code is useful for your research, please cite this paper.



## 🌸 Acknowledgment
Part of our SINet framework is referred to [DSPNet](https://github.com/syc11-25/DSPNet/tree/main). Code and data processing for the pavia dataset: the code [[Hyper-DSNet](https://github.com/liangjiandeng/Hyper-DSNet)] + the dataset  [[HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection)] for fair training and testing! We thank all the contributors for open-sourcing
