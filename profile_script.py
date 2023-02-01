import torch
from cuda.nsection import entmax_nsect_cuda


def main():
    x = torch.randn(10, 10000, device=torch.device("cuda:0"), dtype=torch.float32)
    out = entmax_nsect_cuda(x, alpha = 1.5, n_iter=9, n_sections=4)
    return 0

if __name__ == "__main__":
    main()