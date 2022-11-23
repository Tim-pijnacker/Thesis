#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t prob(scalar_t z, scalar_t tau, scalar_t alpha) {
    z = ((z - tau) < 0)?0:(z - tau);
    // z = fmax((z - tau), 0);
    const auto power = 1 / (alpha - 1);
    return pow(z, power);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sctn_check(
    scalar_t z,
    scalar_t tauLo, 
    scalar_t tauHi, 
    scalar_t alpha,
    int dim
){
    // update fLo in global memory, and then take summ 
    const auto fLo = prob(z, tauLo, alpha); 
    // fLo = sum(fLo, dim) - 1; ??
    const auto fHi = prob(z, tauHi, alpha);
    // fHi = sum(fHi, dim) - 1 ??
    return fLo;
}

torch::Tensor entmax_cuda_forward(
    torch::Tensor Z,
    float alpha,
    int dim,
    int nIters,
    int nSections,
    bool ensure_sum_one
) {
    return test;
}