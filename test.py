import torch
import nsection_cpp


def p(Z, tau, alpha):
    Z = torch.clamp(Z - tau, min=0)
    Z = Z ** (1 / (alpha - 1))
    return Z

Z = torch.randn(4)
tau = torch.randn(4)
a = [1.5]*4
alpha = torch.Tensor(a)
output = nsection_cpp.forward(Z, tau, alpha)
print(output)
print(p(Z, tau, alpha))