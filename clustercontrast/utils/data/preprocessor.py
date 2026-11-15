from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, image_process=None,
                 resolution_process=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.image_process = image_process
        self.resolution_process = resolution_process

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.image_process is not None:
            img = self.image_process(img, fpath)

        if self.resolution_process is not None:
            img = self.resolution_process(img, fpath)

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

class Preprocessor_color(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform1=None,
                 resolution_process1=None):
        super(Preprocessor_color, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1
        self.resolution_process1 = resolution_process1
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_ori = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img_ori)
            second = img_ori
            if self.resolution_process1 is not None:
                second = self.resolution_process1(second, fpath)
            img1 = self.transform1(second)
        return img, img1,fname, pid, camid, index
