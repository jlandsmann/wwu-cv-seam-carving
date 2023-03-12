from constants import DEVICE
from init import init
from train import train
from test import test

def main():
    print(f"Using {DEVICE} device")

    init()
    train()
    #test()
    pass


main()