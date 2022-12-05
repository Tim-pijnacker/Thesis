#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t prob(scalar_t x, scalar_t tau, scalar_t alpha)
{
    const auto clamp = ((x - tau) < 0.0) ? 0.0 : (x - tau);
    return pow(clamp, (1.0 / (alpha - 1.0)));
}

template <typename scalar_t>
__global__ void entmax_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tauLo,
    scalar_t alpha,
    scalar_t tauWidth
){
    const int col = blockIdx.x * threadIdx.x + threadIdx.x;
    const int row = blockIdx.z;
    const int section = blockIdx.y;
    
    if(col < x.size(1)){
        const auto sctnTau = tauLo[row][0] + section*tauWidth;
        p[row][section][col] = prob(x[row][col], sctnTau, alpha);
    }
    
}
} // namespace


torch::Tensor entmax_cuda_forward(
    torch::Tensor x,
    float alpha,
    int nIters,
    int nSections
){
    auto options = torch::TensorOptions().device(torch::device_of(x)).dtype(x.dtype());
    auto shape = torch::_shape_as_tensor(x);
    auto bsz = shape[0].item<int>();
    auto d = shape[1].item<int>();
    auto df = shape[1].item<float>();

    auto max = torch::amax(x, -1, true);
    auto tauLo = max * (alpha - 1.0) - 1.0;
    x = x * (alpha - 1.0);

    df = pow(df, alpha - 1.0);
    auto tauWidth = (df - 1.0)/df;

    // prob matrix for cuda kernel
    auto p = torch::zeros({bsz, nSections, d}, options);

    // vectors for searchsorted and index
    auto obj = torch::empty({bsz,nSections}, options);
    auto res = torch::empty({bsz,1}, options.dtype(torch::kInt64));
    auto onesVec = -torch::ones({bsz,1}, options);
    auto temp = torch::empty({bsz}, options);
    auto zero = torch::zeros({1}, options.dtype(torch::kInt64));
    
    const int threads = 1024;
    const dim3  blocks((d + threads - 1) / threads, nSections, bsz);

    for(int i = 0; i < nIters; i++){
        // torch::zero_out(p, p);

        tauWidth /= (nSections - 1.0);

        AT_DISPATCH_FLOATING_TYPES(x.type(), "lltm_forward_cuda", ([&] {
            entmax_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                p.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                alpha,
                tauWidth
                );
        }));
        // cudaDeviceSynchronize();

        torch::sum_out(obj, p, -1);
        torch::searchsorted_out(res, -obj, onesVec);
    }
    return res;
}