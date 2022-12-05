import torch

import nsection_cuda

torch.manual_seed(42)
x = torch.randn(2,3,device=torch.device("cuda:0"), dtype=torch.float32)
print(x)
print(nsection_cuda.forward(x, 1.5, 1, 2))