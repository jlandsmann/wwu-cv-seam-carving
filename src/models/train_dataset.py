from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from constants import BATCH_SIZE, DATA_ROOT

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)