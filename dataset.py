#!/usr/bin/env python3

from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, root_path, data_dict, transform=None):
        self.root_path = root_path
        self.data_dict = data_dict
        self.image_name = list(data_dict.keys())
        self.pseudo_label = list(data_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        image_name = self.image_name[index]
        pseudo_label = self.pseudo_label[index]

        image = Image.open(os.path.join(self.root_path, image_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, pseudo_label, image_name



