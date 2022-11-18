import torch
from torch.autograd import Function

import nsection_cpp


class EntmaxNsectFunction(Function):
    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=5, n_sections=5, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_val = alpha.item()
        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.alpha_val = alpha_val
        ctx.dim = dim

        p_m = nsection_cpp.forward(X, alpha, alpha_val, dim, n_iter, n_sections, ensure_sum_one)

        ctx.save_for_backward(p_m)
        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        d_alpha = None
        if ctx.alpha_val == 2:
            dX = nsection_cpp.sparsemax_backward(Y, dY, ctx.dim)
            return dX, d_alpha, None, None, None, None

        dX = nsection_cpp.entmax_backward(Y, dY, ctx.alpha, ctx.dim)
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            
            d_alpha = nsection_cpp.alpha_gradient(Y, dY, ctx.alpha, ctx.dim)

        return dX, d_alpha, None, None, None, None


def entmax_nsect_cpp(X, alpha=1.5, dim=-1, n_iter=5, n_sections=5, ensure_sum_one=True):
    return EntmaxNsectFunction.apply(X, alpha, dim, n_iter, n_sections, ensure_sum_one)