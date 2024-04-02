import argparse
import os

import cv2
import einops
import numpy as np
import scipy
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms

from utils.dataset_utils import *
import torch.nn.functional as F


class ImageDataset(Dataset):

    def __init__(self, image_files, labels, n_classes, onehot=True, transform=None):
        self.x = list(image_files)
        self.y = list(labels)
        self.n = n_classes
        self.transform = transform
        self.onehot = onehot

    def __len__(self):
        return len(self.x)

    @staticmethod
    def process_image(f):
        img = plt.imread(f)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, -1)
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        if img.dtype == np.uint8:
            img = img.astype('float32') / 255.
        return img

    def __getitem__(self, idx):
        img = self.process_image(self.x[idx])
        sample = {'x': img}
        if self.transform:
            sample = self.transform(sample)
        if self.onehot:
            sample['y'] = torch.nn.functional.one_hot(torch.tensor(self.y[idx]), self.n).to(torch.float32)
        else:
            sample['y'] = self.y[idx]
        return sample


def __metadata_injector__(s, *a, **kw): return s


class DatasetWrapper(Dataset):

    @staticmethod
    def convert_key(key, inplace):
        return key

    def __init__(self, dataset, inplace=False, metadata=None, metadata_injector=None, injection=0, transform=None,
                 **kwargs):
        if metadata is not None:
            for k in metadata.keys():
                if type(metadata[k]) == np.ndarray:
                    metadata[k] = torch.tensor(metadata[k])
        if metadata_injector is None:
            metadata_injector = __metadata_injector__
        self.dataset = dataset
        self.inplace = inplace
        self.metadata = metadata
        self.metadata_injector = metadata_injector
        self.injection = injection
        self.injection_kwargs = kwargs
        self.operator_kwargs = {}

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        if self.injection < 0:
            sample = self.metadata_injector(sample, self.metadata, **self.injection_kwargs)
        sample = self.operator(sample, **self.operator_kwargs)
        if self.injection > 0:
            sample = self.metadata_injector(sample, self.metadata, **self.injection_kwargs)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def operator(self, sample, **kwargs):
        return sample
