import torch
from cuda.nsection import entmax_nsect_cuda1

entmax_nsect_cuda1(x, alpha = 1.5, n_iter=5, n_sections=32)
