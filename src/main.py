import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from init import init
from models.image_dataset import get_test_dataset
from constants import DEVICE
from train import train

def main():
    print(f"Using {DEVICE} device")

    init()
    train()

    pass


main()