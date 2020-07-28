import h5py
import os
import cv2
import numpy as np
from skimage.transform import resize

def get_img(cfile, shape = (256,256)):
    img = cv2.cvtColor(cv2.imread(cfile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = resize(img, shape)*255
    return img

def get_mask(cfile, shape = (256,256)):
    GT = cv2.imread(cfile, 0)
    GT = resize(GT, shape)
    return GT

img_src = '/media/data1/talshah/GenA/buildings_vaihingen/buildings/img/'
mask_src = '/media/data1/talshah/GenA/buildings_vaihingen/buildings/AllBuildingsMask/'
mask_single_src = '/media/data1/talshah/GenA/buildings_vaihingen/buildings/mask_sizeold/'
hf_tri = h5py.File('/media/data1/talshah/DeepACM/Data/full_training_viah.h5', 'w')
hf_test = h5py.File('/media/data1/talshah/DeepACM/Data/full_test_viah.h5', 'w')
a_img = os.listdir(img_src)
a_img.sort()
a_mask = os.listdir(mask_src)
a_mask.sort()
a_mask_single = os.listdir(mask_single_src)
a_mask_single.sort()

imgs_tri = hf_tri.create_group('imgs')
mask_tri = hf_tri.create_group('mask')
mask_single_tri = hf_tri.create_group('mask_single')

imgs_test = hf_test.create_group('imgs')
mask_test = hf_test.create_group('mask')
mask_single_test = hf_test.create_group('mask_single')
shape = (64, 64)

for folder in a_img[:100]:
    print('training: ' + folder)
    cfile = img_src + folder
    img = get_img(cfile, shape)
    imgs_tri.create_dataset(folder, data=img, dtype=np.uint8)

for folder in a_img[100:]:
    print('validation: ' + folder)
    cfile = img_src + folder
    img = get_img(cfile, shape)
    imgs_test.create_dataset(folder, data=img, dtype=np.uint8)

for folder in a_mask[:100]:
    print('training: ' + folder)
    cfile = mask_src + folder
    mask = get_mask(cfile, shape)
    mask_tri.create_dataset(folder, data=mask, dtype=np.uint8)

for folder in a_mask[100:]:
    print('validation: ' + folder)
    cfile = mask_src + folder
    mask = get_mask(cfile, shape)
    mask_test.create_dataset(folder, data=mask, dtype=np.uint8)

for folder in a_mask_single[:100]:
    print('training: ' + folder)
    cfile = mask_single_src + folder
    mask = get_mask(cfile, shape)
    mask_single_tri.create_dataset(folder, data=mask, dtype=np.uint8)

for folder in a_mask_single[100:]:
    print('validation: ' + folder)
    cfile = mask_single_src + folder
    mask = get_mask(cfile, shape)
    mask_single_test.create_dataset(folder, data=mask, dtype=np.uint8)

hf_tri.close()
hf_test.close()

