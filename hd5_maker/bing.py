import h5py
import os
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms


def get_img(cfile):
    image = Image.open(cfile)
    image = transforms.functional.resize(image, (256, 256), 3)
    return np.asarray(image)

def get_mask(cfile):
    image = Image.open(cfile)
    image = np.asarray(image).copy()
    image[image > 0] = 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image).convert('1')
    mask_one = transforms.functional.resize(image, (256, 256))
    mask = np.asarray(mask_one).astype(np.float)
    return mask

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

