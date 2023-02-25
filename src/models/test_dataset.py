from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from constants import BATCH_SIZE, DATA_ROOT

test_data = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=ToTensor(),
)

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)