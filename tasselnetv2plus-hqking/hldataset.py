"""
@author: hao lu
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd 
import random
import numpy as np
from PIL import Image
import cv2
import h5py
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from skimage import util
from skimage.measure import label
from skimage.measure import regionprops
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


def read_image(x):#读取图片
    img = Image.open(x)
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale//
         img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr,img


class RandomCrop(object):#随机裁剪
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h, w = image.shape[:2]

        if isinstance(self.output_size, tuple):
            new_h = min(self.output_size[0], h)
            new_w = min(self.output_size[1], w)
            assert (new_h, new_w) == self.output_size
        else:
            crop_size = min(self.output_size, h, w)
            assert crop_size == self.output_size
            new_h = new_w = crop_size
        if gtcount > 0:
            mask = target > 0
            ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[ch:h-ch+1, cw:w-cw+1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            if len(idh) != 0:
                ids = random.choice(range(len(idh)))
                hc, wc = idh[ids], idw[ids]
                top, left = hc-ch, wc-cw
            else:
                top = np.random.randint(0, h-new_h+1)
                left = np.random.randint(0, w-new_w+1)
        else:
            top = np.random.randint(0, h-new_h+1)
            left = np.random.randint(0, w-new_w+1)

        image = image[top:top+new_h, left:left+new_w, :]
        target = target[top:top+new_h, left:left+new_w]

        return {'image': image, 'target': target, 'gtcount': gtcount}


class RandomFlip(object):#随机翻转
    def __init__(self):
        pass

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return {'image': image, 'target': target, 'gtcount': gtcount}


class Normalize(object):#归一化

    def __init__(self, scale, mean, std):#按照规定的平均值和方差进行归一化
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']#获取实例的图片，目标位置和目标数量
        image, target = image.astype('float32'), target.astype('float32')

        # pixel normalization 像素层面归一化
        image = (self.scale * image - self.mean) / self.std

        image, target = image.astype('float32'), target.astype('float32')

        return {'image': image, 'target': target, 'gtcount': gtcount}


class ZeroPadding(object):#零填充
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize =  self.psize

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h,w = image.size()[-2:]
        ph,pw = (psize-h%psize),(psize-w%psize)
        # print(ph,pw)

        (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)
        (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
        if (ph!=psize) or (pw!=psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            image = F.pad(image,tmp_pad)
            target = F.pad(target,tmp_pad)

        return {'image': image, 'target': target, 'gtcount': gtcount}


class ToTensor(object):#转换为Tensor数据类型
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis 交换像素变为 通道X高度X宽度
        # numpy image: H x W x C
        # torch image: C X H X W
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image = image.transpose((2, 0, 1))
        target = np.expand_dims(target, axis=2)
        target = target.transpose((2, 0, 1))
        image, target = torch.from_numpy(image), torch.from_numpy(target)
        return {'image': image, 'target': target, 'gtcount': gtcount}


class MaizeTasselDataset(Dataset):#Maize数据集划分
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):#数据路径，数据划分比例
        self.data_dir = data_dir#数据路径
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]#读取每一行的数据 以空格分开
        self.ratio = ratio#裁剪
        self.train = train#表示是否是训练集
        self.transform = transform#是否进行transform
        self.image_list = []
        
        # store images and generate ground truths
        self.images = {}#图片
        self.targets = {}#目标
        self.gtcounts = {}#数量
        self.dotimages = {}#点图

    def bbs2points(self, bbs): #处理标记点boundxox 转为 点
        points = []
        for bb in bbs:
            x, y = [float(b) for b in bb]#左上角坐标+宽和高
            # x2, y2 = x1+w-1, y1+h-1
            # x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)#求中心点
            points.append([x, y])#添加到点集
        return points

    def __len__(self):#返回长度
        return len(self.data_list)

    def __getitem__(self, idx):#获取实例
        file_name = self.data_list[idx]#名称
        self.image_list.append(file_name[0])#将得到的新的图片加入
        if file_name[0] not in self.images:
            image,img = read_image(self.data_dir+file_name[0])
            root = 'MTC/image/train'  # 保存地址
            picture_name = 'IMG_'+str(idx)+'.jpg'
            path = root + "/" + picture_name  # 保存地址
            print(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annotation = sio.loadmat(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(path, image)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if annotation['annotation'][0][0][1] is not None:
                bbs = annotation['annotation'][0][0][1]
                gtcount = bbs.shape[0]
                pts = self.bbs2points(bbs)
                for pt in pts:
                    pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[pt[1], pt[0]] = 1
                    cv2.circle(dotimage, (pt[0], pt[1]), int(24 * self.ratio) , (255, 0, 0), -1)
                count = idx
                output_dir = 'MTC'
                output_file = 'GT_IMG' + '_' + str(count) + '.txt'
                print(output_file)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, 'w') as f:
                    for pt in pts:
                        print(pt)
                        x, y = pt
                        line = f'{x} {y}\n'
                        f.write(line)

            else:
                gtcount = 0
            target = gaussian_filter(target, 80 * self.ratio)

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())

            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})
            self.dotimages.update({file_name[0]:dotimage})

        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class WhearEarDataset(Dataset):#whear数据集
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []
        
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}

    def parsexml(self, xml):
        tree = ET.parse(xml)
        root = tree.getroot()
        bbs = []
        for bb in root.iter('bndbox'):#获取boundbox 坐标位置
            xmin = int(bb.find('xmin').text)
            ymin = int(bb.find('ymin').text)
            xmax = int(bb.find('xmax').text)
            ymax = int(bb.find('ymax').text)
            bbs.append([xmin, ymin, xmax, ymax])
        return bbs

    def bbs2points(self, bbs):   #转化为中心点坐标
        points = []
        for bb in bbs:
            x1, y1, x2, y2 = [float(b) for b in bb]
            x, y = np.round((x1+x2)/2).astype(np.int32), np.round((y1+y2)/2).astype(np.int32)
            points.append([x, y])
        return points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):#获取实例
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image = read_image(self.data_dir+file_name[0])
            bbs = self.parsexml(self.data_dir+file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            dotimage = image.copy()
            if bbs is not None:
                gtcount = len(bbs)
                pts = self.bbs2points(bbs)
                for pt in pts:
                    pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[pt[1], pt[0]] = 1
                    cv2.circle(dotimage, (pt[0], pt[1]), 6, (255, 0, 0), -1)
            else:
                gtcount = 0
            target = gaussian_filter(target, 40 * self.ratio)

            # cmap = plt.cm.get_cmap('jet')
            # target_show = target / (target.max() + 1e-12)
            # target_show = cmap(target_show) * 255.
            # target_show = 0.5 * image + 0.5 * target_show[:, :, 0:3]
            # plt.imshow(target_show.astype(np.uint8))
            # plt.show()
            # print(target.sum())

            # plt.imshow(dotimage.astype(np.uint8))
            # plt.show()

            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})
            self.dotimages.update({file_name[0]:dotimage})

        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SorghumHeadDataset(Dataset):#Sor数据集
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []
        
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}
        self.points = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image = read_image(self.data_dir+file_name[0])
            annotations = read_image(self.data_dir+file_name[1])
            annotations = util.img_as_ubyte(annotations[:, :, 0]) == 0
            annotations = label(annotations, connectivity=annotations.ndim)
            annotations = regionprops(annotations)

            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            points=[]
            dotimage = image.copy()
            if annotations is not None:

                gtcount = len(annotations)
                for annotation in annotations:
                    pt = annotation.centroid
                    x, y = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                    target[x, y] = 1
                    points.append([x,y])
                    cv2.circle(dotimage, (y, x), 6, (255, 0, 0), -1)
                print(points)
                output_dir = 'dataset2_train'
                output_file = 'IMG'+'_'+str(idx)+ '.txt'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, 'w') as f:
                    for point in points:
                        x, y = point
                        line = f'{x} {y}\n'
                        f.write(line)

            else:
                gtcount = 0
            target = gaussian_filter(target, 8 * self.ratio)



            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})
            self.dotimages.update({file_name[0]:dotimage})


        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__=='__main__':

    dataset = MaizeTasselDataset(
        data_dir='./data/maize_counting_dataset',
        data_list='./data/maize_counting_dataset/train.txt',
        ratio=0.125,
        train=True,
        transform=transforms.Compose([
            ToTensor()]
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    mean = 0.
    std = 0.
    for i, data in enumerate(dataloader, 0):
        images, targets= data['image'], data['target']
        bs = images.size(0)
        images = images.view(bs, images.size(1), -1).float()
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataloader)
    std /= len(dataloader)
