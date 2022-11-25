import torch
from torch.autograd import Function, grad
import numpy as np

from py.nsection import entmax_nsect
import nsection_cpp

# test input
torch.manual_seed(42)
x = torch.randn(2,3)
alpha = 1.5

print(nsection_cpp.forward(x, alpha, 5, 5))
print(entmax_nsect(x))
