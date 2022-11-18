import torch
import torch.nn as nn
from torch.autograd import Function


class EntmaxBisectFunction(Function):
    @classmethod
    def p(cls, Z, tau: float, alpha: float):
        Z = torch.clamp(Z - tau, min=0)
        Z = Z ** (1 / (alpha - 1))
        return Z

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X * (alpha - 1)
        max_val = max_val * (alpha - 1)

        # Note: when alpha < 1, tau_lo > tau_hi. This still works since dm < 0.
        tau_lo = max_val - 1
        tau_hi = max_val - ((1 / d) ** (alpha - 1))

        dm = tau_hi - tau_lo
        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls.p(X, tau_m, alpha)
            f_m = p_m.sum(dim) - 1
            mask = (f_m >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)
        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None


def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)


class EntmaxBisect(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=50):
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        return entmax_bisect(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter
        )
