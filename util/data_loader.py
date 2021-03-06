import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch as t
import torch.utils.data
import torchvision as tv


def balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = t.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = t.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


def load_data(dataset, data_dir, batch_size, workers, val_split=0.):
    if val_split < 0 or val_split >= 1:
        raise ValueError('val_split should be in the range of [0, 1) but got %.3f' % val_split)

    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    if dataset == 'imagenet':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'), transform=train_transform)
        test_set = tv.datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'), transform=val_transform)

    elif dataset == 'cifar10':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomGrayscale(),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        test_set = tv.datasets.CIFAR10(data_dir, train=False, transform=val_transform, download=True)

    else:
        raise ValueError('load_data does not support dataset %s' % dataset)

    if val_split != 0:
        train_set, val_set = balance_val_split(train_set, val_split)
    else:
        # In this case, use the test set for validation
        val_set = test_set

    train_loader = t.utils.data.DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = t.utils.data.DataLoader(
        val_set, batch_size, num_workers=workers, pin_memory=True)
    test_loader = t.utils.data.DataLoader(
        test_set, batch_size, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader
