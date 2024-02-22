import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, labels_table_path, img_dir, transform=transforms.ToTensor(), num_classes=3):
        """
        Args:
            labels_table_path (string): Path to the Excel file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_excel(labels_table_path)
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
        label = int(self.img_labels.Label[idx])
        label = np.eye(self.num_classes, self.num_classes, dtype=np.float32)[label, :]    # one-hot encoding
        if self.transform:
            image = self.transform(image)
        return image, label
