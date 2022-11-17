import torch
import nsection_cpp
from torch.autograd import Function
from nsection_bench import EntmaxNsectFunction


def p(Z, tau, alpha):
    Z = torch.clamp(Z - tau, min=0)
    Z = Z ** (1 / (alpha - 1))
    return Z

def sctn_check(X, tau_lo, tau_hi, alpha, dim):
    f_lo = p(X, tau_lo, alpha)
    f_lo = f_lo.sum(dim) - 1
    f_hi = p(X, tau_hi, alpha)
    f_hi = f_hi.sum(dim) - 1
    out = ((f_lo > 0) & (f_hi <= 0)).unsqueeze(dim)
    return out

# test input
Z = torch.randn(10,5)
dim = -1

# test tau lo and hi
tau_lo = torch.randn(10,5)-1
tau_hi = tau_lo+2

# test alpha
alpha = 1.5

print(EntmaxNsectFunction.forward(Function, Z, alpha))
print(nsection_cpp.forward(Z, tau_lo, alpha, dim))