from torch import cuda
from torch.backends import mps

BATCH_SIZE = 64
DATA_ROOT = "data"
DEVICE = "cpu" #"cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"