import torch
from torch.autograd import Function, grad
import numpy as np

from cpp.nsection import entmax_nsect_cpp
from py.nsection import entmax_nsect
from time import time

# test input
torch.manual_seed(42)
Z = torch.randn(50,30000)
alpha = 1.5

