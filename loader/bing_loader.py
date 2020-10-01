import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import loader.transforms as transforms
from PIL import Image


class bing_segmentation(Dataset):
    def __init__(self, ann='training', args=None):
        self.ann = ann
        self.MEAN = np.array([101.87901, 100.81404, 110.389275])
        self.STD = np.array([17.022379, 17.664776, 20.302572])
        if ann == 'training':
            self.transformations = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.ColorJitter(brightness=0.3,
                                                                              contrast=0.3,
                                                                              saturation=0.3,
                                                                              hue=0.01),
                                                       transforms.RandomRotation(90),
                                                       transforms.RandomVerticalFlip(),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(self.MEAN, self.STD)])
        else:
            self.transformations = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(self.MEAN, self.STD)])
        if ann == 'training':
            self.data_length = 335
        else:
            self.data_length = 270
        self.args = args

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        if self.ann == 'training':
            self.data = h5py.File('Data/full_training_Bing.h5', 'r')
        else:
            self.data = h5py.File('Data/full_test_Bing.h5', 'r')
        self.mask = self.data['mask_single']
        self.imgs = self.data['imgs']
        self.img_list = list(self.imgs)
        self.mask_list = list(self.mask)
        cimage = self.img_list[item]
        img = self.imgs.get(cimage).value
        cmask = self.mask_list[item]
        mask = self.mask.get(cmask).value
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        img, mask = self.transformations(img, mask)
        return img, mask
