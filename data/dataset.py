#!/usr/bin/env python3


import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading images and their labels.
    
    Args:
        data_dict (dict): A dictionary containing image names as keys and labels as values.
        root_dir (str): The directory where the images are located.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, data_dict, root_dir, transform=None):
        self.data_dict = data_dict
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data_dict)

    def __getitem__(self, idx):
        """
        Returns one sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image, label) where label is the label of the image.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = list(self.data_dict.keys())[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        label = int(self.data_dict[img_name])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_name

    
