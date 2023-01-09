import torch
import pytest

from python.nsection import entmax_nsect
from cuda.nsection import entmax_nsect_cuda
from cpp.nsection import entmax_nsect_cpp
from entmax import entmax_bisect


# edge cases test
@pytest.mark.parametrize("dim", (1000, 30000))
@pytest.mark.parametrize("device", (torch.device("cuda:0"), torch.device("cpu")))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("alpha", (1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75))
def test_nsection_out(alpha, dtype, device, dim):
    # test input
    torch.manual_seed(40)
    x = torch.randn(10, dim, dtype=dtype, device=device)

    # python nsection entmax 
    y_hat_py = entmax_nsect(x, alpha, n_iter=50, n_sections=2)
    y_hat_cpp = entmax_nsect_cpp(x, alpha, n_iter=50, n_sections=2)
    if device == torch.device("cuda:0"):
        y_hat_cuda = entmax_nsect_cuda(x, alpha, n_iter=50, n_sections=2)

    # python bisection entmax
    y_true = entmax_bisect(x, alpha)

    assert torch.allclose(y_hat_py, y_true)
    assert torch.allclose(y_hat_cpp, y_true)
    if device == torch.device("cuda:0"):
        assert torch.allclose(y_hat_cuda, y_true)

import torch.utils.benchmark as benchmark


torch.manual_seed(40)
x = torch.randn(100, 30000, dtype=torch.float32, device=torch.device("cuda:0"))


out1 = entmax_nsect_cuda(x, 1.75, 50, 2)
out2 = entmax_nsect(x, 1.75, 50, 2)
out3 = entmax_nsect_cpp(x, 1.75, 50, 2)
outtrue = entmax_bisect(x, 1.75)
print("\n Cuda1:")
print(out1)

print("\n Py: ")
print(out2)

print(torch.sum(torch.abs(out1 - outtrue)))
print(torch.sum(torch.abs(out3 - outtrue)))
# timer = benchmark.Timer(
#     stmt='entmax_nsect_cuda(x, alpha = 1.5, n_iter=5, n_sections=32)',
#     setup='from cuda.nsection import entmax_nsect_cuda',
#     globals={'x': x},
#     num_threads=1,
#     label="cuda comparison",
#     sub_label="[10, 1000]",
#     description="cuda",
# ).timeit(100)

# timer1 = benchmark.Timer(
#     stmt='entmax_nsect_cuda1(x, alpha = 1.5, n_iter=5, n_sections=32)',
#     setup='from cuda.nsection import entmax_nsect_cuda1',
#     globals={'x': x},
#     num_threads=1,
#     label="cuda comparison",
#     sub_label="[10, 1000]",
#     description="cuda1",
# ).timeit(100)

# compare = benchmark.Compare([timer, timer1])
# compare.colorize()
# compare.print()