import torch
from torch.autograd import Function


class EntmaxNsectFunction(Function):
    @classmethod
    def p(cls, Z, tau: float, alpha: float):
        Z = torch.clamp(Z - tau, min=0)
        Z = Z ** (1 / (alpha - 1))
        return Z

    @classmethod
    def sctn_check(cls, X, tau_lo: float, tau_hi: float, alpha: float, dim):
        f_lo = cls.p(X, tau_lo, alpha)
        f_lo = f_lo.sum(dim) - 1
        f_hi = cls.p(X, tau_hi, alpha)
        f_hi = f_hi.sum(dim) - 1
        out = ((f_lo > 0) & (f_hi <= 0)).unsqueeze(dim)
        return out

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True, n_sections=5):

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

        sctn_rng = tau_hi - tau_lo
        for it in range(n_iter):

            sctn_rng /= n_sections

            for sctn_id in range(1,n_sections+1):
                sctn_tau_lo = tau_lo + ((sctn_id - 1)*sctn_rng)
                sctn_tau_hi = tau_lo + (sctn_id*sctn_rng)
                mask = cls.sctn_check(X, sctn_tau_lo, sctn_tau_hi, alpha, dim)
                tau_lo = torch.where(mask, sctn_tau_lo, tau_lo)
        
        p_m = cls.p(X, tau_lo, alpha)
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
