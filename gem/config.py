import os

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

K_LAYERS = 7
D_DYN = 3
TAU_SELF = 0.07
TAU_DYN = 20.0
WS_STATIC = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.9]
W_VERB, W_OBJECT, W_ACTION = 0.2, 0.2, 0.6

DEFAULT_MAX_FRAMES = 16
SAMPLE_STRIDE = 1
SEED = 1234

CMAP = "jet"
ALPHA = 0.45
