from torch import cuda
from torch.backends import mps

MODEL_NAME = "CNN-SGD-4"
MODEL_PATH = "data/models/" + MODEL_NAME + ".pth"
TRAIN_ITEMS = 512000
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
MOMENTUM= 99e-2
BETA_RANGE = (0.9, 0.999)
EPISLON = 1e-08
DATA_ROOT = "../gecarvteBilder"
DEVICE = "cpu" # "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"