# Model Refinement on TasselNetv2+ 模型改进

<p align="center">
  <img src="计数结果示例.jpg" width="800"/>
</p>
基于陆昊老师的TasselNetv2+植物计数网络。<br>
本章中提出了一些方法，试图从算法类型，网络结构，参数大小三个角度出发，寻找提升模型性能的方法。<br>   
由于原模型对在高粱穗数据集上的表现已经较好，对改进模型的训练主要在玉米穗数据集和麦穗数据集上进行。<br><br>

注：以下模型改进中的训练结果可在Model Rfinement on Maize和Model Rfinement on Wheat文件夹中查看。<br>
其中部分基于麦穗数据集的训练在Kaggle上完成，结果不便保存。<br>
如有任何疑问，意见或建议，请联系张祎程'U202015006@hust.edu.cn'。<br><br>


## 改变算法类型
### 改变激活函数
原模型卷积层激活函数为ReLU，考虑将其更换为LeakyReLU是否能提升模型性能。<br><br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.19, best mse: 9.13, best_rmae: 37.67, best_rmse: 82.45, best_r2: 0.8861
````
与原模型相比，mae指标从5.48下降到5.19，mse指标从10.06下降到9.13，性能有一定的提升。<br><br>
在麦穗数据集上的训练结果：<br>
````
best mae: 5.93, best mse: 7.81, best_rmae: 4.39, best_rmse: 5.68, best_r2: 0.8612
````
很遗憾，各项性能指标都变差了。<br><br>
结论：更换激活函数为LeakyReLU能一定程度提升模型在玉米穗数据集上的性能。<br><br>


### 改变损失函数
原模型以torch.nn.L1loss即平均绝对误差为损失函数，考虑更换损失函数是否能提升模型性能。<br><br>
机器学习中，损失函数(Loss Function)用于定义单个训练样本与真实值之间的误差。训练模型的目的即是最小化损失函数。<br>
平均绝对误差(Mean Absolute Error, MAE)，也称L1loss，为目标值与预测值之差绝对值和的均值，<br>
均方误差(Mean Squared Error, MSE)，也称L2loss，为目标值与预测值之差平方和的均值的二分之一,<br>
平滑L1损失(Smooth L1 Loss, SLL)，优化的平均绝对误差。<br><br>
采用MSE作为损失函数，在玉米穗数据集上的训练结果：<br>
````
best mae: 5.90, best mse: 9.64, best_rmae: 44.87, best_rmse: 103.97, best_r2: 0.8776
````
性能一定程度上变差。<br><br>
在麦穗数据集上训练的结果：<br>
````
best mae: 4.47, best mse: 5.39, best_rmae: 3.42, best_rmse: 4.14, best_r2: 0.9212
````
与原模型相比，mae和rmae指标基本相同，mse和rmse指标有一定的提升。<br><br>
结论：更换损失函数为MAE能一定程度提升模型在麦穗数据集上的性能。<br><br><br>
采用SLL作为损失函数，在玉米穗数据集上的训练结果：<br>
````
best mae: 6.06, best mse: 10.59, best_rmae: 40.69, best_rmse: 81.99, best_r2: 0.8449
````
在麦穗数据集上的训练结果：<br>
````
best mae: 4.57, best mse: 5.67, best_rmae: 3.43, best_rmse: 4.21, best_r2: 0.9163
````
与原模型相比，在两个数据集上的性能指标都下降。<br><br>
结论：采用SLL作为损失函数无法提升模型性能。<br><br>


### 改变优化器
原模型默认采用的优化器为SGD，但还提供了另一种方案Adam，考虑更换优化器是否能提升模型性能。<br>
Adam和SGD都是常用的优化器，各有优缺点，具体选择应该根据具体问题和数据集的情况来决定。<br><br>
Adam在玉米穗数据集上的训练结果：<br>
````
best mae: 5.16, best mse: 8.74, best_rmae: 33.98, best_rmse: 80.98, best_r2: 0.8983
````
性能有不错的提升。<br><br>
Adam在麦穗数据集上的训练结果：<br>
````
best mae: 4.68, best mse: 5.81, best_rmae: 3.60, best_rmse: 4.52, best_r2: 0.9102
````
性能略有下降。<br><br>
结论：Adam在玉米穗数据集上表现更优，SGD在麦穗数据集上表现更优。<br><br>

## 改变网络结构
### 改变池化层类型
原模型采用的池化均为最大值池化，参考他人的建议，考虑将第一个最大值池化更换为平均值池化能否提升模型性能。<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.29, best mse: 9.47, best_rmae: 34.07, best_rmse: 80.53, best_r2: 0.8792
````
性能相比原模型有一定的提升。<br><br>
在麦穗数据集上的训练结果：<br>
````
best mae: 5.03, best mse: 6.33, best_rmae: 3.77, best_rmse: 4.76, best_r2: 0.8875
````
性能略有下降。<br><br>
结论：将第一层最大值池化改为平均值池化，在玉米穗数据集上的性能提升，在麦穗数据集上的性能略有下降。<br><br>

### 增加/减少卷积层数
原模型在编码器部分共有5层卷积模型，考虑增加或减少卷积网络层数能否提升模型性能。<br>
增加一层卷积层。参数量增大，运行速度有一定的下降。<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.68, best mse: 10.35, best_rmae: 27.43, best_rmse: 46.52, best_r2: 0.8653
````
在麦穗数据集上的训练结果：<br>
````
best mae: 6.09, best mse: 7.65, best_rmae: 4.54, best_rmse: 5.66, best_r2: 0.9156
````
结论：增加一层卷积层，两个数据集上的各项性能指标均变差，可能是模型层数过多导致过拟合，模型性能下降。<br><br>
减少一层卷积层。参数量减少，运行速度有一定的上升。<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.49, best mse: 9.79, best_rmae: 31.82, best_rmse: 62.63, best_r2: 0.8777
````
与原模型相比，mae基本一致，mse略有提升。<br>
在麦穗数据集上的训练结果：<br>
````
best mae: 6.38, best mse: 7.62, best_rmae: 4.80, best_rmse: 5.79, best_r2: 0.8955
````
结论：减少一层卷积层，在玉米穗数据集上性能略有提升，在麦穗数据集上各项性能指标均变差，可能是网络层数过少，拟合程度不够，模型性能下降。<br><br>

### 改变池化层位置
考虑改变第三次池化的位置能否提升系统的性能。<br>
在玉米穗数据集上训练的结果：<br>
````
best mae: 5.58, best mse: 9.98, best_rmae: 31.94, best_rmse: 59.80, best_r2: 0.8626
````
在麦穗数据集上训练的结果：<br>
````
best mae: 4.99, best mse: 6.22, best_rmae: 3.73, best_rmse: 4.60, best_r2: 0.8967
````
性能均下降。<br>
结论：改变池化层位置，模型性能变差。<br><br>

## 改变模型参数
### 改变动量大小
原模型优化器默认采用SGD算法，其中使用的动量Momentum=0.95。<br>
查阅资料得知动量取值在0.8~0.95之间可能获得最佳的性能。考虑改变动量能否提升模型性能，以麦穗数据集为例进行训练。<br><br>
增大动量，令Momentum=0.98：<br>
````
best mae: 4.31, best mse: 5.34, best_rmae: 3.28, best_rmse: 4.08, best_r2: 0.9263
````
性能指标有一定的提升。<br><br>
减小动量，令Momentum=0.9：<br>
````
best mae: 4.37, best mse: 5.49, best_rmae: 3.35, best_rmse: 4.27, best_r2: 0.9245
````
各项性能指标均有提升，模型性能得到改善。<br><br>
进一步减小动量，令Momentum=0.85：<br>
````
best mae: 4.27, best mse: 5.18, best_rmae: 3.28, best_rmse: 4.01, best_r2: 0.9284
````
各项性能进一步提升，模型性能进一步改善。<br><br>
结论：改变动量进行若干次训练，在Momentum=0.85时取得最好的性能指标结果。相比原模型Momentum=0.95时性能得到了一定的提升。<br><br>
将上述结论在玉米穗数据集上进行训练，取Momentum=0.85：<br>
````
best mae: 5.05, best mse: 9.55, best_rmae: 28.90, best_rmse: 60.02, best_r2: 0.8766
````
性能同样有不错的提升。<br>

### 改变学习率
原模型优化器默认采用SGD算法，其中使用的学习率Learning_Rate=0.01，考虑改变学习率能否提升模型性能。<br>
增大学习率，取Learning_Rate=0.012：<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.86, best mse: 10.48, best_rmae: 36.26, best_rmse: 77.01, best_r2: 0.8548
````
模型性能略有下降。<br><br>
在麦穗数据集上的训练结果：<br>
````
best mae: 4.68, best mse: 5.83, best_rmae: 3.53, best_rmse: 4.38, best_r2: 0.9150
````
模型性能略有下降。<br><br>
减小学习率，取Learning_Rate=0.008：<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.86, best mse: 10.48, best_rmae: 36.26, best_rmse: 77.01, best_r2: 0.8548
````
模型性能略有下降。<br><br>
在麦穗数据集上的训练结果：<br>
````
best mae: 5.50, best mse: 6.65, best_rmae: 4.16, best_rmse: 4.97, best_r2: 0.9044
````
模型性能下降。<br><br>

### 改变衰减权重
原模型优化器默认采用SGD算法，其中使用的权重衰减Weight_Decay=0.0005，考虑改变权重衰减能否提升模型性能。<br>
增大衰减权重，取Weight_Decay=0.0008：<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.16, best mse: 8.86, best_rmae: 28.25, best_rmse: 54.67, best_r2: 0.8915
````
模型性能有一定的提升。<br><br>
在麦穗数据集上的训练结果：<br>
````
best mae: 5.97, best mse: 7.57, best_rmae: 4.50, best_rmse: 5.65, best_r2: 0.8966
````
模型性能下降。<br><br>
减小衰减权重，取Weight_Decay=0.0002:<br>
在玉米穗数据集上的训练结果：<br>
````
best mae: 5.31, best mse: 8.98, best_rmae: 35.83, best_rmse: 89.62, best_r2: 0.8938
````
模型性能有一定的提升。<br><br>
在麦穗数据集上的训练结果：<br>
````
best mae: 6.22, best mse: 7.66, best_rmae: 4.66, best_rmse: 5.75, best_r2: 0.8780
````
模型性能下降。<br><br>

### 改变批次大小
原模型采用Batch_Size=9,考虑改变批次大小能否提升模型性能。以麦穗数据集为例进行训练。<br>
增大批次大小，令Btach_Size=10:<br>
````
best mae: 4.65, best mse: 5.51, best_rmae: 3.53, best_rmse: 4.22, best_r2: 0.9251
````
增大批次大小，理论上减少训练时间，增加系统稳定但，但实测性能下降，可能无法继续优化模型性能。<br><br>
减小批次大小，令Btach_Size=8:<br>
````
best mae: 4.36, best mse: 5.42, best_rmae: 3.34, best_rmse: 4.21, best_r2: 0.9226
````
各项指标均有一定的提升，模型性能得到改善。<br><br>
进一步减少批次大小，令Btach_Size=7:<br>
````
best mae: 4.69, best mse: 5.78, best_rmae: 3.61, best_rmse: 4.47, best_r2: 0.9160
````
各项指标均变差，模型性能较初始下降，再减小批次大小可能无法提升性能。<br><br>
结论：改变若干次批次大小进行训练，在Btach_Size=8获得了最佳的性能。<br><br>
将上述结论在玉米穗数据集上进行训练，取Btach_Size=8：<br>
````
best mae: 5.20, best mse: 9.27, best_rmae: 29.98, best_rmse: 62.74, best_r2: 0.8862
````
模型性能同样有一定的提升。<br><br>


### 改变随机种子数
原模型初始化时采用的随机种子数为2020，考虑改变随机种子数能否提升模型性能。<br>
参考他人建议，将随机种子数设置为3407：<br>
````
best mae: 5.54, best mse: 6.73, best_rmae: 4.22, best_rmse: 5.18, best_r2: 0.8795
````
模型性能略有下降。<br><br>
将随机种子数设置为1500：<br>
````
best mae: 5.05, best mse: 6.32, best_rmae: 3.76, best_rmse: 4.66, best_r2: 0.9162
````
模型性能略有下降。<br><br>
选取不同的种子数，模型表现出来的性能稍有不同。虽然未能达到优化模型的目的，但体现出模型具有较好地鲁棒性。<br><br>

## 对模型的综合改进
综合考虑以上多种模型改进的方法。<br>
对于玉米穗数据集，考虑将激活函数ReLU替换为LeakyReLU，采用MSE作为损失函数，以Adam为优化器，将第一层最大值池化改为平均值池化，批次大小Batch_Size=8，其余与原模型保持一致。
获得最终训练效果如下：<br>
````
best mae: 4.92, best mse: 9.06, best_rmae: 36.67, best_rmse: 82.96, best_r2: 0.8882
````
原模型与改进模型指标对比如下表所示：<br>

 指标  | MAE  | MSE | RMAE | RMSE | R^2
 ---- | ----- | ------ | ------- | -------- | --------- |
  原模型 | 5.48 | 10.06 | 27.65 | 51.65 | 0.8607 |
 改进模型  | 4.92 | 9.06 | 36.67 | 82.96 | 0.8882 |

与原模型相比，mae指标从5.48下降到4.92，mse指标从10.06下降到9.06，相关系数增大，性能有不错的提升，提升程度约10.22%。<br><br>
对于麦穗数据集，考虑使用动量Momentum=0.85，批次大小Batch_Size=8，其余与原模型保持一致。<br>
获得的最终训练效果如下：<br>
````
best mae: 4.21, best mse: 5.14, best_rmae: 3.25, best_rmse: 4.11, best_r2: 0.9295
````
原模型与改进模型指标对比如下表所示：<br>

 指标  | MAE  | MSE | RMAE | RMSE | R^2
 ---- | ----- | ------ | ------- | -------- | --------- |
  原模型 | 4.44 | 5.41 | 3.42 | 4.19 | 0.9288 |
 改进模型  | 4.21 | 5.14 | 3.25 | 4.11 | 0.9295 |
 
与原模型相比，与原模型相比，mae指标从4.44下降到4.21，mse指标从5.41下降到5.14，性能有不错的提升，提升程度约4.51%。<br><br>
