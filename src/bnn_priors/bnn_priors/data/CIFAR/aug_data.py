from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
  
# copied from elsewhere in the repo

class CIFAR10Aug(CIFAR10):
    '''Wrapper class to return original image besides augmentations.
    '''
    K = 1

    def __init__(self, n_aug=1, base_transform=None, aug_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.n_aug = n_aug
        self.aug_dir = aug_dir
        self.base_transform = base_transform

    def __getitem__(self, index):
        img, target = Image.fromarray(self.data[index]), self.targets[index]

        aug_imgs = None
        if self.aug_dir is not None:
            aug_imgs = torch.load(f'{self.aug_dir}/{index}.pt')
        elif self.transform is not None:
            aug_imgs = torch.stack([self.transform(img) for _ in range(self.n_aug)])
        if self.base_transform is not None:
            img = self.base_transform(img)

        return img, [target, aug_imgs]

