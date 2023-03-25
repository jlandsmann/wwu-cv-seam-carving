from torch import cuda
from torch.backends import mps

MODEL_NAME = "CNN-ADAM-3"
MODEL_PATH = "data/models/" + MODEL_NAME + ".pth"
TRAIN_ITEMS = 512000
BATCH_SIZE = 24
EPOCHS = 10
LEARNING_RATE = 1e-2
MOMENTUM= 1e-1
BETA_RANGE = (0.9, 0.999)
EPISLON = 1e-03
DATA_ROOT = "../gecarvteBilder"
DEVICE = "cpu" # "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"