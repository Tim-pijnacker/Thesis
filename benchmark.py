import torch

# functions to benchmark
from python.nsection import entmax_nsect
from cpp.nsection import entmax_nsect_cpp
from entmax import sparsemax_bisect, entmax_bisect, entmax15

import timeit

#input for benchmarking
cuda_device = torch.device("cuda:0")
inp = torch.randn(10, 5)
inp_cuda = torch.randn(10, 5, device=cuda_device)
x = inp


print(entmax_nsect(x))
print(entmax_nsect_cpp(x))


# Ensure that both functions compute the same output
# assert entmax_nsect(x).allclose(entmax_nsect_cpp(x))
# assert torch.sum((entmax_nsect(x) - entmax_nsect_cpp(x)) ** 2) < 1e-7

# t0 = timeit.Timer(
#     stmt='entmax_nsect(x)',
#     setup='from python.nsection import entmax_nsect',
#     globals={'x': x})

# t1 = timeit.Timer(
#     stmt='entmax_nsect_cpp(x)',
#     setup='from cpp.nsection import entmax_nsect_cpp',
#     globals={'x': x})

# # print(f'entmax_nsect(x):        {t0.timeit(100) / 100 * 1e6:>5.1f} us')
# # print(f'entmax_nsect_cpp(x):    {t1.timeit(100) / 100 * 1e6:>5.1f} us')
# print(f'entmax_nsect(x):        {t0.timeit(100)}')
# print(f'entmax_nsect_cpp(x):    {t1.timeit(100)}')