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