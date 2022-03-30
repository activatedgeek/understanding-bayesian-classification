import random
import numpy as np
import torch

def set_seeds(seed=None):
  if seed is not None and seed >= 0:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
