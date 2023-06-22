# 人群计数模型P2PNet

这个仓库包含了两部分内容：<br>
* 一是基于人群计数模型**P2PNet**迁移到植物计数数据集上的代码，论文见[Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework](https://arxiv.org/abs/2107.12746).<br>
* 二是将**Global Wheat Head Detection Dataset**转换为适用于TasselNetv2+模型和P2PNet模型的代码，论文见[Global Wheat Head Detection (GWHD) dataset: a large and diverse dataset of high resolution RGB labelled images to develop and benchmark wheat head detection methods](https://arxiv.org/abs/2005.02162).
 

## P2PNet的可视化测试结果
<img src="vis/pred12.jpg"/> 

## 网络结构
P2PNet的整体结构如下。在VGG16的基础上，首先引入了一个上采样路径来获得细粒度的特征图。
然后，利用两个分支来同时预测一组点和它们的置信度分数。

<img src="vis/net.png" width="1000"/>   

## Wheat Ears Counting Dataset.
| Method        | MAE   | MSE   |
| ------------- | ----- | ----- |
| TasselNetv2+  | 4.44  | 5.41  |
| P2PNet        | 4.0   | 5.40  |

可以看出在Wheat Ears Counting Dataset上P2PNet的性能优于 TasselNetv2+.

## Maize Tassels Counting Dataset.
| Method        | MAE   | MSE   |
| ------------- | ----- | ----- |
| TasselNetv2+  | 5.48  | 10.06 |
| P2PNet        | 9.19  | 9.38  |

可以看出在Maize Tassels Counting Dataset上P2PNet的性能要比 TasselNetv2+差.

## Global Wheat Head Detection Dataset.
通过该仓库中
| Method        | MAE   | MSE   |
| ------------- | ----- | ----- |
| TasselNetv2+  | 5.48  | 10.06 |
| P2PNet        | 9.19  | 9.38  |

可以看出在Maize Tassels Counting Dataset上P2PNet的性能要比 TasselNetv2+差.
