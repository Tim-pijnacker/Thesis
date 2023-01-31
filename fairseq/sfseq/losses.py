import torch
import torch.nn as nn
from torch.autograd import Function
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/home/timpijnacker/Thesis/cuda/')


from nsection import entmax_nsect_cuda
# from python.nsection import entmax_nsect
# from cpp.nsection import entmax_nsect_cpp



class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        assert reduction in ["elementwise_mean", "sum", "none"]
        self.reduction = reduction
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):
        loss = self.loss(X, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "elementwise_mean":
            loss = loss.sum() / size
        return loss


class _GenericLossFunction(Function):
    @classmethod
    def forward(cls, ctx, X, target, alpha, proj_args):
        """
        X (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert X.shape[0] == target.shape[0]

        p_star = cls.project(X, alpha, **proj_args)
        loss = cls.omega(p_star, alpha)

        p_star.scatter_add_(1, target.unsqueeze(1), torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, X)
        ctx.save_for_backward(p_star)

        return loss

    @classmethod
    def backward(cls, ctx, grad_output):
        p_star, = ctx.saved_tensors
        grad = grad_output.unsqueeze(1) * p_star
        ret = (grad,)

        # pad with as many Nones as needed
        return ret + (None,) * (1 + cls.n_fwd_args)



class NsectCudaLossFunction(_GenericLossFunction):

    n_fwd_args = 1

    @classmethod
    def project(cls, X, alpha, k=None):
        return entmax_nsect_cuda(X, alpha=alpha, n_iter=9, n_sections=4)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star * torch.sqrt(p_star)).sum(dim=1)) / 0.75

    @classmethod
    def forward(cls, ctx, X, target, k=None):
        return super().forward(ctx, X, target, alpha=1.5, proj_args=dict(k=k))



def NsectCuda_loss(X, target):
    """1.5-entmax loss: sparse alternative to cross-entropy
    Computed using a partial sorting strategy.
    Parameters
    ----------
    X : torch.Tensor, shape=(n_samples, n_classes)
        The input 2D tensor of predicted scores
    target : torch.LongTensor, shape=(n_samples,)
        The ground truth labels, 0 <= target < n_classes.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return NsectCudaLossFunction.apply(X, target)


class NsectCudaLoss(_GenericLoss):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        super(NsectCudaLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return NsectCuda_loss(X, target)