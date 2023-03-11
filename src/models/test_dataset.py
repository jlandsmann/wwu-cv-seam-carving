from torch.utils.data import DataLoader

from constants import BATCH_SIZE, DATA_ROOT
from .image_dataset import ImageDataset

test_data = ImageDataset(annotations_file=DATA_ROOT + "/test/labels.csv", img_dir=DATA_ROOT + "/test")
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)