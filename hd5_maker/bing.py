import h5py
import os
import cv2
import numpy as np
from skimage.transform import resize
from glob import glob

def get_img(cfile):
    img = cv2.cvtColor(cv2.imread(cfile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = resize(img, (64, 64))*255
    return img

def get_mask(cfile):
    GT = cv2.imread(cfile, 0)
    GT = resize(GT, (64, 64))
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0
    return GT

src = '/media/data1/talshah/DAR/single_buildings/'
hf_tri = h5py.File('/media/data1/talshah/DeepACM/Data/full_training_Bing.h5', 'w')
hf_test = h5py.File('/media/data1/talshah/DeepACM/Data/full_test_Bing.h5', 'w')
a = os.listdir(src)
a.sort()

img_list = glob(src + 'building_*')
mask_list = glob(src + 'building_mask_*')
mask_all_list = glob(src + 'building_mask_all_*')
img_list.sort()
mask_list.sort()
mask_all_list.sort()

imgs_tri = hf_tri.create_group('imgs')
mask_tri = hf_tri.create_group('mask')
mask_single_tri = hf_tri.create_group('mask_single')

imgs_test = hf_test.create_group('imgs')
mask_test = hf_test.create_group('mask')
mask_single_test = hf_test.create_group('mask_single')

for folder in img_list[0:335]:
    print('training: ' + folder)
    img = get_img(folder)
    imgs_tri.create_dataset(folder.split('/')[-1], data=img, dtype=np.uint8)

for folder in img_list[335:606]:
    print('validation: ' + folder)
    img = get_img(folder)
    imgs_test.create_dataset(folder.split('/')[-1], data=img, dtype=np.uint8)

for folder in mask_list[0:335]:
    print('training: ' + folder)
    mask = get_mask(folder)
    mask_single_tri.create_dataset(folder.split('/')[-1], data=mask, dtype=np.uint8)

for folder in mask_list[335:606]:
    print('validation: ' + folder)
    mask = get_mask(folder)
    mask_single_test.create_dataset(folder.split('/')[-1], data=mask, dtype=np.uint8)

for folder in mask_all_list[0:335]:
    print('training: ' + folder)
    mask = get_mask(folder)
    mask_tri.create_dataset(folder.split('/')[-1], data=mask, dtype=np.uint8)

for folder in mask_all_list[335:606]:
    print('validation: ' + folder)
    mask = get_mask(folder)
    mask_test.create_dataset(folder.split('/')[-1], data=mask, dtype=np.uint8)

hf_tri.close()
hf_test.close()

