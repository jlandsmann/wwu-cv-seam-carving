import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from init import init
from models.image_dataset import get_test_dataset
from constants import DEVICE
from train import train


def output_test_images():
    data = get_test_dataset()
    dataloader = DataLoader(data, batch_size=64, shuffle=False)

    features, labels = next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = features[0].squeeze().permute(1, 2, 0)
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def main():
    print(f"Using {DEVICE} device")

    output_test_images()

    pass


main()