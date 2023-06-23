# PR-2023
TasselNetv2+: A Fast Implementation for High-Throughput Plant Counting from High-Resolution RGB Imagery

## 一、修改部分在CPU环境下运行代码会无法支持的代码，并对代码添加注释,由于代码过多，只列出部分，详情参考所提交的代码
***gen_trainval_list.py***

```python
import os
import glob
import random
root = './data/wheat_ears_counting_dataset'#文件目录
image_folder = 'images'#目录下保存图片的目录
label_folder = 'labels'#标签
train = 'train'
val = 'val'

train_path = os.path.join(root, train)#训练路径
with open('train.txt', 'w') as f:#打开该文件
    for image_path in glob.glob(os.path.join(train_path, image_folder, '*.JPG')):#读取图片
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.JPG', '.xml')
        f.write(im_path+'\t'+gt_path+'\n')

val_path = os.path.join(root, val)
with open('val.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(val_path, image_folder, '*.JPG')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.JPG', '.xml')
        f.write(im_path+'\t'+gt_path+'\n')

```
## 二、添加早停
早停法的核心思想是在训练过程中检验模型在验证数据上的表现，一旦验证损失停止减小（或者连续几轮未明显减小），就停止训练。

early_stop
```python
# -*- coding = utf-8 -*-
# @TIME: 2023/4/23 19:21
# @Author :hqKing
# @File : early_stop.py
import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

```
## 三、修复部分bug
在haldataset代码中关于maize数据集的代码在读取bbs时存在bug，标注为点集，然而源代码中的处理是把标注当为boundbox来处理了，导致读取格式错误，修改后如下
```python

    def bbs2points(self, bbs): 
        points = []
        for bb in bbs:
            x, y = [float(b) for b in bb]#标注就是点，无需转换
            # x2, y2 = x1+w-1, y1+h-1
            # x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)#求中心点
            points.append([x, y])#添加到点集
        return points
    
```
## 四、训练开源模型
###CSRNet
CSRNet是一种数据驱动的深度学习方法，可以理解高度拥挤的场景，进行精确的计数估计，并提供高质量的密度图
CSRNet由两个主要部分组成：一个是作为二维特征提取的前端卷积神经网络（即模型中的前端网络frontend ），另一个是用于后端的扩展CNN（后端网络backend），它使用扩展的核（空洞卷积操作）来传递更大的感受野，并代替池化操作，在目前主流的人群计数模型中一般都是使用密度图来呈现的，然而，如何生成准确的密度分布图是一个挑战。一个主要的困难来自于预测方式：由于生成的密度值遵循逐像素的预测，因此输出的密度图必须包含空间相关性，以便能够呈现最近像素之间的平滑过渡，CSRnet提出的改进为设计一个基于CNN的密度图生成器。模型使用纯卷积层作为主干来支持具有灵活分辨率的输入图像。为了限制网络的复杂度，在所有层中使用小尺寸的卷积滤波器（如3×3）。我们将VGG-16[21]的前10层作为前端，将空洞卷积层（dilated convolution layers）作为后端，以扩大感受野并在不丢失分辨率的情况下提取更深层的特征（因为不使用池化层），使用该架构跑出的效果达到了当时的soat

<div align=center>
<img src="https://github.com/hqking0424/PR-2023/blob/whq/1.png"/>
</div>




