import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import loader.transforms as transforms
from PIL import Image


class viah_segmentation(Dataset):
    def __init__(self, ann='training', args=None):
        self.ann = ann
        self.MEAN = np.array([0.47341759*255, 0.28791303*255, 0.2850705*255])
        self.STD = np.array([0.22645572*255, 0.15276193*255, 0.140702*255])
        if ann == 'training':
            self.transformations = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.RandomResizedCrop(size=(256, 256),
                                                                                    mask_size=(256, 256),
                                                                                    scale=(0.75, 1.5)),
                                                       transforms.ColorJitter(brightness=0.4,
                                                                              contrast=0.4,
                                                                              saturation=0.4,
                                                                              hue=0.1),
                                                       transforms.RandomRotation(25),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(self.MEAN, self.STD)])
        else:
            self.transformations = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(self.MEAN, self.STD)])
        if ann == 'training':
            self.data_length = 100
        else:
            self.data_length = 68
        self.args = args

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        if self.ann == 'training':
            self.data = h5py.File('Data/full_training_viah.h5', 'r')
        else:
            self.data = h5py.File('Data/full_test_viah.h5', 'r')
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

