from torch import cuda
from torch.backends import mps

MODEL_NAME = "DCNN-ADAM-3"
MODEL_PATH = "data/models/" + MODEL_NAME + ".pth"
OFFSET = 0
TRAIN_ITEMS = 512000
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
BETA_RANGE = (0.9, 0.999)
EPISLON = 1e-04
DATA_ROOT = "../gecarvteBilder"
DEVICE = "cpu" # "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"