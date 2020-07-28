import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import cv2
import random
import numpy as np
from skimage.transform import AffineTransform, warp
from skimage.util import random_noise
from PIL import Image
from skimage.filters import gaussian


class viah_segmentation(Dataset):
    def __init__(self, ann='training', is_aug=True, args=None):
        self.ann = ann
        self.MEAN = np.array([0.47341759, 0.28791303, 0.2850705])
        self.STD = np.array([0.22645572, 0.15276193, 0.140702])
        if ann=='training':
            self.transformations_img = transforms.Compose([
                                                           transforms.ColorJitter(brightness=0.6, contrast=0.5,
                                                                                  saturation=0.4, hue=0.025),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(self.MEAN, self.STD)])
        else:
            self.transformations_img = transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize(self.MEAN, self.STD)])
        self.transformations_mask = transforms.Compose([transforms.ToTensor()])
        self.is_aug = is_aug
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

        if self.is_aug:
            a = np.random.randint(0, 2)
            if a==1:
                Idim = int(self.args['dim'])
                half = int(Idim/2)
                k = random.choice([0, 15, 45, 60, 90, 135, 180, 210, 240, 270])
                m = random.choice([0.75, 1.0, 1.25, 1.5])
                rot_mat = cv2.getRotationMatrix2D((half, half), k, m)
                img = cv2.warpAffine(img, rot_mat, (Idim, Idim))
                mask = cv2.warpAffine(mask, rot_mat, (Idim, Idim))

            a = np.random.randint(0, 3)
            if a == 1:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if a == 2:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

            a = np.random.randint(0, 3)
            if a == 1:
                sigma = np.random.randint(0, 10)/100
                img = random_noise(img/255, var=sigma ** 2)*255
                pass

        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        img = Image.fromarray(img)
        img = self.transformations_img(img)
        mask = self.transformations_mask(mask)*255
        return img, mask

