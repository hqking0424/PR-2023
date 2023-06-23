import  h5py
import  scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm as CM
from image import *

# root is the path to ShanghaiTech dataset
root='./data1/'

part_B_train = os.path.join(root,'whe/train','image')
part_B_test = os.path.join(root,'whe/test','image')
path_sets = [part_B_train,part_B_test]


img_paths  = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for  img_path  in img_paths:
    print (img_path)
    matpath2=img_path.replace('image', 'ground_truth')
    matpath1= matpath2.replace('IMG_','GT_IMG_')
    mat_path =matpath1.replace('.JPG','.mat')
    print(mat_path)

    mat = io.loadmat(mat_path)
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter(k, 15)
    h5_path1 = img_path.replace('.JPG', '.h5')
    h5_path=h5_path1.replace('image','ground_truth')
    print(h5_path)
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter(k, 15)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
        hf['density'] = k

