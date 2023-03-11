import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from constants import TRAIN_ITEMS, OFFSET


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return min(len(self.img_labels) - OFFSET, TRAIN_ITEMS)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx + OFFSET, 0])
        image = read_image(img_path, ImageReadMode.GRAY).float()
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
