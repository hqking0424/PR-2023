# MCNN多列卷积神经网络运行植物计数数据集


# 模型原理
1.MCNN克服了先前的模型中存在的运行条件苛刻，人群密集情况下运行性能较差的问题。

2.MCNN突破了原本的模型算法，采用了在没有背景分割的基础上实现了人群计数的方法。

3.与此同时，MCNN包含了三列卷积网络，每一列卷积网络分别学习不同感受野的特征。

# 运行过程
1.下载好植物计数的数据集。

2.按照MCNN的格式，重命名图片格式为IMG_图片记号。

3.修改切割图片的matlab文件，将图片数量修改为对应的图片数量。



# 训练
1.对MCNN的代码进行修改，使得能够稳定运行。

2. 运行train.py

3. 校验产生的损失函数结果。


# Other notes
1. During training, the best model is chosen using error on the validation set. (It is not clear how the authors in the original implementation choose the best model).
2. 10% of the training set is set asised for validation. The validation set is chosen randomly.
3. The ground truth density maps are obtained using simple gaussian maps unlike the original method described in the paper.
4. Following are the results on  Shanghai Tech A and B dataset:
		
                |     |  MAE  |   MSE  |
                ------------------------
                | A   |  110  |   169  |
                ------------------------
                | B   |   25  |    44  |
		
5. Also, please take a look at our new work on crowd counting using cascaded cnn and high-level prior (https://github.com/svishwa/crowdcount-cascaded-mtl),  which has improved results as compared to this work. 
               

