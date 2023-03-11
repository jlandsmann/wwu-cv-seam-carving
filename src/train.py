import torch
from torch import nn
from torch.utils.data import DataLoader

from models.neural_network import NeuralNetwork
from models.test_dataset import test_dataloader
from models.train_dataset import train_dataloader
from constants import DEVICE, LEARNING_RATE, EPOCHS, BATCH_SIZE, MODEL_PATH

def do_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def do_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train():
    model = NeuralNetwork()
    model.load_state_dict(torch.load(MODEL_PATH))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        do_train(train_dataloader, model, loss_fn, optimizer)
        do_test(test_dataloader, model, loss_fn)

        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved PyTorch Model State to " + MODEL_PATH)
    print("Done Training!")