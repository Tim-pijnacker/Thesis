import torch
from torch.autograd import Function, grad
import numpy as np

from cpp.nsection import entmax_nsect_cpp
from py.nsection import entmax_nsect
from entmax import entmax_bisect
from time import time

# test input
torch.manual_seed(42)
Z = torch.randn(10,5)
alpha = 1.5

print(entmax_nsect_cpp(Z)[0])
print(entmax_bisect(Z)[0])