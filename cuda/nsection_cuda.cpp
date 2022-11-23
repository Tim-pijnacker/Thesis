#include <torch/extension.h>
#include <vector>

// cuda declarations

torch::Tensor entmax_cuda_forward();

// torch::Tensor entmax_cuda_backward();

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor nsection_forward(
    torch::Tensor Z,
    float alpha,
    int dim,
    int nIters,
    int nSections,
    bool ensure_sum_one
){
    CHECK_INPUT(Z);

    return entmax_cuda_forward(Z, alpha, alphaVal, dim, nIters, nSections, ensure_sum_one);
}