import torch
from cuda.nsection import entmax_nsect_cuda1


def main():
    x = torch.randn(100, 10000, device=torch.device("cuda:0"), dtype=torch.float32)
    out = entmax_nsect_cuda1(x, alpha = 1.5, n_iter=9, n_sections=4)
    return 0

if __name__ == "__main__":
    main()