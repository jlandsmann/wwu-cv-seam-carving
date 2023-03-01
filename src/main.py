import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from init import init
from constants import DEVICE
from train import train

def main():
    print(f"Using {DEVICE} device")

    init()
    train()

    pass


main()