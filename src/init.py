import torch

from models.neural_network import NeuralNetwork
from constants import DEVICE

def init():
    model = NeuralNetwork().to(DEVICE)
    print(model)

    torch.save(model.state_dict(), "data/models/v1.pth")
    print("Saved PyTorch Model State to data/models/v1.pth")
