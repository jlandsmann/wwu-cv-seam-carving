import torch
from models.neural_network import NeuralNetwork
from models.test_dataset import test_dataloader
from constants import MODEL_PATH, DEVICE, BATCH_SIZE

def test():
    model = NeuralNetwork()
    model.load_state_dict(torch.load(MODEL_PATH))

    size = len(test_dataloader.dataset)
    correct = 0
    model.eval()
    print("Test Error: \n")
    with torch.no_grad():
        for idx, (X, y) in enumerate(test_dataloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if (idx % 5 == 0):
                current = (idx + 1) * BATCH_SIZE
                currently_correct = correct
                currently_correct /= current
                print(f"Accuracy: {(100*currently_correct):>0.1f}% [{current:>5d}/{size:>5d}]")
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}% \n")