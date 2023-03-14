import torch
from os.path import exists
from time import sleep

from models.cnn import CNN
from constants import DEVICE, MODEL_PATH

def init():
    if (exists(MODEL_PATH)):
        print("Model already exists. Skipping initiation.")
        print("Using existing model and starting in 5s")
        sleep(5)
        return

    model = CNN().to(DEVICE)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved PyTorch Model State to " + MODEL_PATH)
