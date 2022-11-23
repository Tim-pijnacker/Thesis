import torch
import torch.nn as nn
from torch.autograd import Function

import nsection_cpp


class EntmaxNsectFunction(Function):
    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=5, n_sections=5, ensure_sum_one=True):
        ctx.alpha = alpha
        ctx.dim = dim

        p_m = nsection_cpp.forward(X, alpha, dim, n_iter, n_sections, ensure_sum_one)

        ctx.save_for_backward(p_m)
        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        if ctx.alpha == 2:
            dX = nsection_cpp.sparsemax_backward(Y, dY, ctx.dim)
        else:
            dX = nsection_cpp.entmax_backward(Y, dY, ctx.alpha, ctx.dim)

        return dX, None, None, None, None, None


def entmax_nsect_cpp(X, alpha=1.5, dim=-1, n_iter=5, n_sections=5, ensure_sum_one=True):
    return EntmaxNsectFunction.apply(X, alpha, dim, n_iter, n_sections, ensure_sum_one)


class EntmaxNsect(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=5, n_sections=5):
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        self.n_sections = n_sections
        super().__init__()

    def forward(self, X):
        return entmax_nsect_cpp(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter, n_sections=self.n_sections
        )