#utils.py
import os
import random
import yaml
import numpy as np
import torch

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

