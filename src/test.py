import torch
from models.neural_network import NeuralNetwork
from models.test_dataset import test_data

def test():
    model = NeuralNetwork()
    model.load_state_dict(torch.load("data/models/v1.pth"))

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')