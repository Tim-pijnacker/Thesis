import torch
from torch.autograd import Function, grad
import numpy as np

from nsection import entmax_nsect_cpp
from nsection_bench import entmax_nsect
from time import time

# test input
# torch.manual_seed
Z = torch.randn(50,30000)
alpha = 1.5

start = time()
p = entmax_nsect_cpp(Z, alpha)
print(time() - start)

start = time()
p = entmax_nsect(Z, alpha)
print(time() - start)