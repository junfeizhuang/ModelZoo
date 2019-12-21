import torch
import cv2
import os
import numpy as np
import random

from glob import glob
from torch.utils.data.dataset import Dataset

class TinyImageNet(Dataset):
    def __init__(self, trans, txt):
        with open(txt, 'r') as f:
            self.images_labels = [x.rstrip() for x in f.readlines()]
        self.num = len(self.images_labels)
        self.trans = trans

    def __getitem__(self, idx):
        assert idx < self.num, 'image index overflow'
        image = self.images_labels[idx].split(' ')[0]
        label = int(self.images_labels[idx].split(' ')[1])
        img = cv2.imread(image)
        img = self.trans(img)
        img = np.array(img)
        return img, label

    def __len__(self):
        return self.num
        