from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from constants import BATCH_SIZE, DATA_ROOT
from .image_dataset import ImageDataset

training_data = ImageDataset(annotations_file="data/train/labels.csv", img_dir="data/train")
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)