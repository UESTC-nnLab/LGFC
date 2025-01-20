# LGFC

Local-Global Feature Collaborative Learning with Level-Wise Decoding for Infrared Small Target Detection

Currently, it is still in the process of refinement. After the formal publication of the paper, the code will be further improved.
![frame](https://github.com/user-attachments/assets/475b242e-6300-4ebd-9315-5d6364ce760d)


## Introduction

Due to low signal-to-noise ratio and weak vision contrast, infrared small targets are usually prone to be overwhelmed by backgrounds. Therefore, avoiding losing target information and acquiring distinctive features between targets and backgrounds are pretty crucial. However, existing methods generally rely on convolutions and transformers independently. They cannot effectively capture robust target features, especially in complex scenes. To address this issue, we propose a new artificial intelligence (AI) scheme, local-global feature collaborative learning (LGFC). It could adequately integrate local coarse features and global context cooperatively. Specifically, we design a global \textit{Gaussian-mask Vision Transformer} group with \textit{Global Gaussian Attention} and \textit{Local Window Attention} to obtain refined global features. Then, the local coarse features acquired by a convolution encoder coordinate with refined global features via \textit{Local-Global Collaborating}. Besides, to avoid feature loss in decoding, we devise level-wise decoding with \emph{Cross-layer Feature Interaction} to retain target information in deep networks. Additionally, considering the contours of infrared small targets, we perform post-processing to obtain more accurate detection results by \emph{Coarse-to-Fine Refine}. The experiments on two public datasets show the superiority and generalization ability of this AI-based LGFC for infrared small target detection over state-of-the-art methods, almost 2.3\% higher than the best F1 on each dataset. 

## Datasets

- Datasets are available at [NUAA-SIRST](https://github.com/YimianDai/sirst) and [IRSTD-1K](https://github.com/RuiZhang97/ISNet)

## Prerequisite

- python == 3.8
- pytorch == 1.10.0
- einops == 0.7.0
- opencv-python == 4.7.0.72
- scikit-learn == 1.2.2
- scipy == 1.9.1
- Tested on Ubuntu 20.04.6, with CUDA 12.0, and 1x NVIDIA 3090(24 GB)

## ROC curves
![roc](https://github.com/user-attachments/assets/f93a5940-12ad-495d-9786-ccad58e3642c)

## Contact

IF any questions, please contact with Weiwei Duan via email: [dwwuestc@163.com]().

## Reference

- Li, Boyang, et al. "Dense nested attention network for infrared small target detection." IEEE Transactions on Image Processing 32 (2022): 1745-1758.
- Lin, Jian, et al. "IR-TransDet: Infrared Dim and Small Target Detection With IR-Transformer." IEEE Transactions on Geoscience and Remote Sensing (2023).
