import torch

from init import init
from test import test
from train import train

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    train();
    pass

main();