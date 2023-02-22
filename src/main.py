import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from init import init
from test import test
from models.image_dataset import getTestDataset
from train import train

def outputTestImages():
    data = getTestDataset();
    dataloader = DataLoader(data, batch_size=64, shuffle=True)

    features, labels = next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = features[0].squeeze()
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    outputTestImages();

    pass

main();