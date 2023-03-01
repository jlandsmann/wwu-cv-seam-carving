from torch.utils.data import DataLoader

from constants import BATCH_SIZE
from .image_dataset import ImageDataset

test_data = ImageDataset(annotations_file="data/test/labels.csv", img_dir="data/test")
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)