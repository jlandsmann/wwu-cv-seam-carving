import torch

from models.neural_network import NeuralNetwork

def init():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = NeuralNetwork().to(device)
    print(model)

    torch.save(model.state_dict(), "data/models/v1.pth")
    print("Saved PyTorch Model State to data/models/v1.pth")
