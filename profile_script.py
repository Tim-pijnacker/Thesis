import torch
from cuda.nsection import entmax_nsect_cuda1


x = torch.randn(10, 100, device=torch.device("cuda:0"), dtype=torch.float32)
entmax_nsect_cuda1(x, alpha = 1.5, n_iter=5, n_sections=32)
