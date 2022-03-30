from pathlib import Path
from PIL import Image

import torch
import numpy as np

from torch.utils.data import random_split, Dataset
from torch.distributions import Categorical
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as transforms

from .augmentations import augmentations, augmentations_all

_CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

_CIFAR_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

_TINY_IMAGENET_TRAIN_TRANSFORM = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
])

_TINY_IMAGENET_TEST_TRANSFORM = transforms.Compose([
    # transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
])


# this and AugMixDataset copied from https://github.com/google-research/augmix/blob/master/cifar.py
def aug(image, preprocess, mixture_width=3, mixture_depth=1, aug_severity=3):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    mixture_width: Number of augmentation chains to mix per augmented example
    mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
    aug_severity: Severity of base augmentation operators
  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations
  # if args.all_ops:
  #   aug_list = augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def prepare_transforms(train_data, augment="std"):
    if augment == "augmix":
        test_transform = _CIFAR_TEST_TRANSFORM
        # test_transform.transforms.pop(0)
        train_data.transform.transforms.pop(-1)
        train_data.transform.transforms.pop(-1)
        dataset = AugMixDataset(train_data, preprocess=test_transform, no_jsd=True)
        return dataset

    if augment == "std":
        transform = _CIFAR_TRAIN_TRANSFORM
    elif augment == "flips":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif augment == "vflips":
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif augment == "crops":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_data.transform = transform
    return train_data


class WrapperDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
    
    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, __value):
        return setattr(self.dataset, 'targets', __value)

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, __value):
        return setattr(self.dataset, 'transform', __value)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class LabelNoiseDataset(WrapperDataset):
    def __init__(self, dataset, n_labels=10, label_noise=0):
        super().__init__(dataset)

        self.C = n_labels

        if label_noise > 0:
            orig_targets = self.targets
            self.noisy_targets = torch.where(
                torch.rand(len(orig_targets)) < label_noise,
                Categorical(probs=torch.ones(self.C) / self.C).sample(torch.Size([len(orig_targets)])),
                torch.Tensor(orig_targets).long())

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        y = self.noisy_targets[i]
        return X, y


class AugmentedDataset(WrapperDataset):
    def __init__(self, dataset, base_transform=None, n_aug=1):
        super().__init__(dataset)

        self.n_aug = n_aug
        self.base_transform = base_transform

    def __getitem__(self, i):
        orig_transform = self.transform
        
        self.transform = None
        X_orig, y = self.dataset[i]
        if self.base_transform is not None:
            X_orig = self.base_transform(X_orig)

        self.transform = orig_transform
        X_augs = torch.stack([self.dataset[i][0] for _ in range(self.n_aug)])

        return X_orig, X_augs, y


class CIFAR10FixedAug(CIFAR10):
    '''Wrapper class to return original image + augmentations as a single dataset.

    Use only via get_cifar10_with_aug.
    '''
    def __init__(self, aug_dir=None, **kwargs):
        assert aug_dir is not None

        aug_dir = Path(aug_dir)

        super().__init__(**kwargs)

        self.aug_data = []
        self.aug_targets = []

        for aug_f in aug_dir.rglob('*.pt'):
            x_aug = torch.load(aug_f).permute(0, 2, 3, 1)
            self.aug_data.append(x_aug)

            idx = int(str(Path(aug_f).name).split('.')[0])
            y = self.targets[idx]
            self.aug_targets.extend([y] * x_aug.shape[0])

            if torch.rand(1).item() > .5:
                break

        self.aug_data = torch.vstack(self.aug_data)
        
        assert len(self.aug_data) == len(self.aug_targets)

    def __len__(self) -> int:
        return super().__len__() + len(self.aug_data)

    def __getitem__(self, index):
        index = (index + self.__len__()) % self.__len__()
        if index < super().__len__():
            return super().__getitem__(index)
        
        index = index - super().__len__()
        return self.aug_data[index].permute(2, 0, 1), self.aug_targets[index]


def train_test_split(dataset, val_size=.1, seed=None):
    N = len(dataset)
    N_test = int(val_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(dataset, [N, N_test], 
                                   generator=torch.Generator().manual_seed(seed))
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test


def get_cifar10(root=None, label_noise=0, augment=True, n_aug=1, return_orig=False):
    train_data = CIFAR10(root=root, train=True, download=True,
                         transform=_CIFAR_TRAIN_TRANSFORM if augment else _CIFAR_TEST_TRANSFORM)
    if label_noise > 0:
        train_data = LabelNoiseDataset(train_data, n_labels=10, label_noise=label_noise)
    if augment and return_orig:
        train_data = AugmentedDataset(train_data, base_transform=_CIFAR_TEST_TRANSFORM,
                                      n_aug=n_aug)

    setattr(train_data, 'total_augs', 9 * 9 * 2)
    setattr(train_data, 'total_classes', 10)

    test_data = CIFAR10(root=root, train=False, download=True,
                        transform=_CIFAR_TEST_TRANSFORM)

    return train_data, test_data


def get_tiny_imagenet(root=None, label_noise=0, augment=True, n_aug=1, return_orig=False):
    train_data = ImageFolder(root=Path(root) / 'tiny-imagenet-200' / 'train',
                             transform=_TINY_IMAGENET_TRAIN_TRANSFORM if augment else _TINY_IMAGENET_TEST_TRANSFORM)
    if label_noise > 0:
        train_data = LabelNoiseDataset(train_data, n_labels=200, label_noise=label_noise)
    if augment and return_orig:
        train_data = AugmentedDataset(train_data, base_transform=_TINY_IMAGENET_TEST_TRANSFORM,
                                      n_aug=n_aug)    
    
    setattr(train_data, 'total_augs', 20 * 2)
    setattr(train_data, 'total_classes', 200)

    val_data = ImageFolder(root=Path(root) / 'tiny-imagenet-200' / 'val', transform=_TINY_IMAGENET_TEST_TRANSFORM)
    
    ## NOTE: Folder not in the right format.
    # test_data = ImageFolder(root=Path(root) / 'tiny-imagenet-200' / 'test', transform=_TINY_IMAGENET_TEST_TRANSFORM)

    return train_data, val_data


def get_cifar10_fixed_aug(root=None, val_size=0, seed=None, aug_dir=None):
    train_data = CIFAR10FixedAug(root=root, train=True, download=True,
                                 aug_dir=aug_dir, transform=_CIFAR_TEST_TRANSFORM)

    test_data = CIFAR10(root=root, train=False, download=True,
                        transform=_CIFAR_TEST_TRANSFORM)

    if val_size != 0:
            train_data, val_data = train_test_split(train_data, val_size=val_size, seed=seed)
            return train_data, val_data, test_data

    return train_data, test_data
