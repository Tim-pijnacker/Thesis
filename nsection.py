import torch
from torch.autograd import Function


def kernel(tau, X, alpha, dummy_mat, sctn_id, dim=-1):
    f = torch.clamp(X - tau, min=0) ** (1 / (alpha - 1))
    f = f.sum(dim) - 1
    mask = (f <= 0).unsqueeze(dim)
    dummy_mat[:, sctn_id] = torch.where(mask, 1.0, 0.0).squeeze()


class EntmaxNsectFunction(Function):
    @classmethod
    def p(cls, Z, tau: float, alpha: float):
        Z = torch.clamp(Z - tau, min=0)
        Z = Z ** (1 / (alpha - 1))
        return Z

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
        tau_hi = max_val - (1 / d) ** (alpha - 1)

        for it in range(n_iter):
            # using tau_lo and tau_max calculate tau_m for each section
            total_rng = tau_hi - tau_lo
            # calculate the range for each section
            sctn_tau_rng = total_rng / n_sections
            sctn_ids = torch.tensor(list(range(1, n_sections)))
            # make a matrix with the tau for each section as columns
            sctn_tau = tau_lo + sctn_tau_rng * sctn_ids

            # run kernel 1 that will place a 1 in the dummy matrix [i,j],
            # if for row i, section j has a negative value
            dummy_mat = torch.zeros(max_val.shape[0], n_sections - 1)
            for sctn_id in range(n_sections - 1):
                tau = sctn_tau[:, sctn_id].unsqueeze(dim=dim)
                kernel(tau, X, alpha, dummy_mat, sctn_id)

            # loop trough dummy matrix to determine how to update tau_lo and tau_hi
            for row_id, row in enumerate(dummy_mat):
                # if there is no 1 in the row, new section is between the last tau and tau_hi
                if 1 not in row:
                    tau_lo[row_id] = sctn_tau[row_id, -1]
                # if the first value in row is 1 then new section between tau_lo and first tau
                elif row[0] == 1:
                    tau_hi[row_id] = sctn_tau[row_id, 0]
                else:
                    for val_id, val in enumerate(row):
                        if val == 1:
                            tau_lo[row_id] = sctn_tau[row_id, val_id - 1]
                            tau_hi[row_id] = sctn_tau[row_id, val_id]
                            break
        
        print(tau)
        p_m = cls.p(X, tau, alpha)
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

