import torch
import pytest

from python.nsection import entmax_nsect
from cpp.nsection import entmax_nsect_cpp
from entmax import entmax_bisect, sparsemax_bisect
import nsection_cuda

# pytest for gpu and cuda?
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("alpha", (1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3))
def test_nsection_out(alpha, dtype):
    # test input
    torch.manual_seed(42)
    x = torch.randn(100,30000, dtype=dtype)

    # python nsection entmax 
    y_hat_py = entmax_nsect(x, alpha, n_iter=15, n_sections=10)
    y_hat_cpp = entmax_nsect_cpp(x, alpha, n_iter=15, n_sections=10)

    # python bisection entmax
    y_true = entmax_bisect(x, alpha)

    assert torch.sum((y_hat_py - y_true) ** 2) < 1e-7
    assert torch.sum((y_hat_cpp - y_true) ** 2) < 1e-7

torch.manual_seed(42)
x = torch.randn(10,5,device=torch.device("cuda:0"), dtype=torch.float32)
out_cuda = nsection_cuda.forward(x, 1.5, 5, 5)
out_py = entmax_nsect(x, 1.5, 5, 5)

print(out_cuda)
print(out_py)