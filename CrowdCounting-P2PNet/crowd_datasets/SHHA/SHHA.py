import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io


class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):

        # data_root: 数据集根目录
        # transform: 数据预处理的变换操作
        # train: 是否为训练模式的布尔值
        # patch: 是否进行图像裁剪的布尔值
        # flip: 是否进行图像翻转的布尔值

        self.root_path = data_root
        self.train_lists = "mtc_train.list"
        self.eval_list = "mtc_test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # 加载图像/标注文件对，将图像列表文件中的图像路径和标注路径进行映射，并将图像路径存储在img_list中。
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # print(img_path)
        # print(gt_path)
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        # print(point.ndim)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.train and self.patch:
            if point.size == 0:
                img, point = img, point
            else:
                img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        # print(img.ndim)
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            if img.ndim == 4:
                img = torch.Tensor(img[:, :, :, ::-1].copy())
            else:
                img = img
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将BGR格式转换为RGB格式
    # 将图像转换为PIL图像对象，方便后续处理

    # 加载标注点
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    return img, np.array(points)  # 返回加载的图像和标注点，标注点以NumPy数组的形式返回


# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])  # 第一维表示裁剪的图像块的数量，
                                                                      # 第二维表示图像的通道数，第三和第四维表示每个图像块的高度和宽度。
                                                                      # 所有元素都被初始化为零，用于存储裁剪后的图像块。
    result_den = []  # 存储裁剪后的点集
    # 对每张图像进行num_patch次裁剪
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)  # 随机生成裁剪起始点的垂直位置
        start_w = random.randint(0, img.size(2) - half_w)  # 随机生成裁剪起始点的水平位置
        end_h = start_h + half_h  # 计算裁剪结束点的垂直位置
        end_w = start_w + half_w  # 计算裁剪结束点的水平位置
        # 复制裁剪后的矩形区域
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # 复制裁剪后的点集
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)  #(den[:, 0] >= start_w) & (den[:, 0] <= end_w) 选择了 x 坐标在 start_w 和 end_w 之间的密度点，(den[:, 1] >= start_h) & (den[:, 1] <= end_h) 选择了 y 坐标在 start_h 和 end_h 之间的密度点。通过逻辑与运算 & 结合这些条件，得到最终的 idx，它是一个布尔数组，与 den 的行数相同，标识了裁剪后的图像区域内哪些密度点被选中。
        # 将坐标进行偏移，将密度点的坐标从裁剪后的图像区域内转换回原始图像的坐标系
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den