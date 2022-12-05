import torch
from torch.autograd import Function


class EntmaxNsectFunction(Function):
    @classmethod
    def forward(cls, ctx, x, alpha, n_iter, n_sections):
        bsz, d = x.shape
        x_max, _ = x.max(dim=-1, keepdim=True)
        x = x * (alpha - 1)
        tau_lo = x_max * (alpha - 1) - 1

        d = d ** (alpha - 1) 
        tau_width = (d - 1) / d  
        tau_frac = torch.linspace(0, 1, n_sections, device=x.device)

        for it in range(n_iter):

            # generate sections
            taus = tau_lo + tau_width * tau_frac

            # compute all ps in one go
            ps = torch.clamp(x.unsqueeze(dim=-2) - taus.unsqueeze(dim=-1), min=0)
            ps = ps ** (1/(alpha - 1))
            
            # compute normalization objective: will be decreasing.
            obj = ps.sum(dim=-1)
            res = torch.searchsorted(-obj, -torch.ones(x.shape[:-1] + (1,), device=x.device))
            res = res.squeeze()

            # tau_hi = taus[torch.arange(bsz), res]  # unnecessary
            tau_lo = taus[torch.arange(bsz), torch.clamp(res - 1, min=0)]
            tau_lo = tau_lo.unsqueeze(-1)
            tau_width /= n_sections

        p = torch.clamp(x - tau_lo, min=0) ** (1/(alpha - 1))
        p /= p.sum(dim=-1).unsqueeze(dim=-1)
        return p

    
def sparsemax_nsection(x, n_iter=5, n_sections=5):
    bsz, d = x.shape
    x_max, _ = x.max(dim=-1, keepdim=True)
    tau_lo = x_max - 1

    tau_width = (d - 1) / d  # = tau_hi - tau_lo
    tau_frac = torch.linspace(0, 1, n_sections)

    for it in range(n_iter):

        # generate sections

        taus = tau_lo + tau_width * tau_frac

        # compute all ps in one go
        ps = torch.clamp(x.unsqueeze(dim=-2) - taus.unsqueeze(dim=-1), min=0)
        
        # compute normalization objective: will be decreasing.
        obj = ps.sum(dim=-1)

        res = torch.searchsorted(-obj, -torch.ones(x.shape[:-1] + (1,)))
        res = res.squeeze()

        # tau_hi = taus[torch.arange(bsz), res]  # unnecessary
        tau_lo = taus[torch.arange(bsz), torch.clamp(res - 1, min=0)]
        tau_lo = tau_lo.unsqueeze(-1)
        tau_width /= n_sections

    return torch.clamp(x - tau_lo, min=0)

def entmax_nsect(x, alpha=1.5, n_iter=5, n_sections=5):
    return EntmaxNsectFunction.apply(x, alpha, n_iter, n_sections)