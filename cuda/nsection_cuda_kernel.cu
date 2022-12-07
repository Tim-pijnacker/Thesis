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
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.z;
    const int section = blockIdx.y;
    
    if(col < x.size(1)){
        const auto sctnTau = tauLo[row][0] + section*tauWidth;
        p[row][section][col] = prob(x[row][col], sctnTau, alpha);
    }
    
}

template <typename scalar_t>
__global__ void entmax_cuda_tauLo_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> res,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tauLo,
    scalar_t tauWidth
){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index < res.size(0)){
        tauLo[index][0] = tauLo[index][0] + (res[index][0] - 1.0)*tauWidth;
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
    auto resf = torch::empty({bsz,1}, options);
    auto onesVec = -torch::ones({bsz,1}, options);
    auto temp = torch::empty({bsz}, options);
    auto zero = torch::zeros({1}, options.dtype(torch::kInt64));
    
    const int threads = 1024;
    const dim3  blocks((d + threads - 1) / threads, nSections, bsz);
    const dim3  blocks2((bsz + threads - 1) / threads, 1, 1);

    for(int i = 0; i < nIters; i++){
        tauWidth /= (nSections);

        AT_DISPATCH_FLOATING_TYPES(x.type(), "lltm_forward_cuda", ([&] {
            entmax_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                p.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                alpha,
                tauWidth
                );
        }));
        cudaDeviceSynchronize();

        torch::sum_out(obj, p, -1);
        torch::searchsorted_out(res, -obj, onesVec, false, true);
        resf = res.to(x.dtype());

        AT_DISPATCH_FLOATING_TYPES(x.type(), "lltm_tauLo_cuda", ([&] {
            entmax_cuda_tauLo_kernel<scalar_t><<<blocks2, threads>>>(
                resf.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                tauWidth
                );
        }));
        cudaDeviceSynchronize();
    }
    auto z = torch::clamp_min(x - tauLo, 0.0);
    auto pOut = torch::float_power(z, 1.0/(alpha - 1.0));
    return pOut.to(x.dtype());
}