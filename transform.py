#!/usr/bin/env python3

import torchvision.transforms as transforms


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_transforms(name="train", norm="0.5", size=299):
    img_size = size
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "0.5":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    if name == "val":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif name == "train":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.8, scale=(0.02, 0.20), ratio=(0.5, 2.0), inplace=True),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        raise NotImplementedError
