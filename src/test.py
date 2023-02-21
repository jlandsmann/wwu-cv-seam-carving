import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from models.neural_network import NeuralNetwork

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

def test():
    model = NeuralNetwork()
    model.load_state_dict(torch.load("data/models/v1.pth"))

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')