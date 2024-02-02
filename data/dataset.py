#!/usr/bin/env python3

from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dict, root_dir, transform=None):
        self.data_dict = data_dict
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        image_name = list(self.data_dict.keys())[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')  # Assuming RGB images

        pseudo_label = int(self.data_dict[image_name])  # Convert label to int if needed

        if self.transform:
            image = self.transform(image)

        return image, pseudo_label, image_name



