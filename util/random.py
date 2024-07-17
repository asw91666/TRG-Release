import numpy as np
import torch

def sample_uniform_distribution(range=(-1,1), size=1):
    return np.random.uniform(range[0], range[1], size)

def sample_uniform_distribution_torch(range=10, size=1):
    return (torch.rand(size) * range * 2) - range