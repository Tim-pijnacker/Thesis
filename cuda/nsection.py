import torch
import torch.nn as nn
from torch.autograd import Function

import nsection_cuda


class EntmaxNsectFunction(Function):
    @classmethod
    def forward(cls, ctx, X, alpha=1.5, n_iter=5, n_sections=5, ensure_sum_one=True):
        ctx.alpha = alpha

        p_m = nsection_cuda.forward(X, alpha, n_iter, n_sections)

        ctx.save_for_backward(p_m)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=-1).unsqueeze(dim=-1)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        if ctx.alpha == 2:
            dX = nsection_cuda.sparsemax_backward(Y, dY)
        else:
            dX = nsection_cuda.entmax_backward(Y, dY, ctx.alpha)

        return dX, None, None, None, None, None


def entmax_nsect_cuda(X, alpha=1.5, n_iter=5, n_sections=5):
    return EntmaxNsectFunction.apply(X, alpha, n_iter, n_sections)


class EntmaxNsect(nn.Module):
    def __init__(self, alpha=1.5, n_iter=5, n_sections=5):
        self.n_iter = n_iter
        self.alpha = alpha
        self.n_sections = n_sections
        super().__init__()

    def forward(self, X):
        return entmax_nsect_cuda(
            X, alpha=self.alpha, n_iter=self.n_iter, n_sections=self.n_sections
        )