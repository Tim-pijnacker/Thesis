import torch
from entmax import sparsemax


def sparsemax_nsection(x, n_iter=5, n_sections=3):
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
        t = x.unsqueeze(dim=-2) - taus.unsqueeze(dim=-1)
        print("-----------------")
        print(x)
        print(x.unsqueeze(dim=-2))
        print("-----------------")
        print(taus)
        print(taus.unsqueeze(dim=-1))
        print("-----------------")
        print(t)
        return "test"
        # compute normalization objective: will be decreasing.
        obj = ps.sum(dim=-1)

        res = torch.searchsorted(-obj, -torch.ones(x.shape[:-1] + (1,)))
        res = res.squeeze()

        # tau_hi = taus[torch.arange(bsz), res]  # unnecessary
        tau_lo = taus[torch.arange(bsz), torch.clamp(res - 1, min=0)]
        tau_lo = tau_lo.unsqueeze(-1)
        tau_width /= n_sections

    return torch.clamp(x - tau_lo, min=0)


def main():

    x = torch.randn(2, 5)
    z = sparsemax_nsection(x, n_sections=10)
    # print(sparsemax(x))
    # print(sparsemax_nsection(x, n_sections=10))


if __name__ == '__main__':
    main()

