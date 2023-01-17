#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

namespace {
template <unsigned int blockSize>
__device__ __forceinline__ void warpReduceSum(volatile float* shmem_ptr, int tid){
    // these if statement are evaluated at compile time
    if (blockSize >= 64) shmem_ptr[tid] += shmem_ptr[tid + 32];
    if (blockSize >= 32) shmem_ptr[tid] += shmem_ptr[tid + 16];
    if (blockSize >= 16) shmem_ptr[tid] += shmem_ptr[tid + 8];
    if (blockSize >=  8) shmem_ptr[tid] += shmem_ptr[tid + 4];
    if (blockSize >=  4) shmem_ptr[tid] += shmem_ptr[tid + 2];
    if (blockSize >=  2) shmem_ptr[tid] += shmem_ptr[tid + 1];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t prob(scalar_t x, scalar_t tau, scalar_t alpha)
{
    if (x < tau){
        return 0.0;
    }
    return powf((x - tau), (1.0 / (alpha - 1.0)));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t prob_add(scalar_t x1, scalar_t x2, scalar_t tau, scalar_t alpha)
{
    const float power = 1.0 / (alpha - 1.0);
    if (x1 < tau && x2 < tau){
        return 0.0;
    } 
    if (x1 < tau){
        return powf((x2 - tau), power);
    }
    if (x2 < tau){
        return powf((x1 - tau), power);
    }
    return powf((x1 - tau), power) + powf((x2 - tau), power);
}

template <typename scalar_t, unsigned int blockSize>
__global__ void p_reduction_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tauLo,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> blockSum,
    scalar_t alpha,
    scalar_t tauWidth
){
    const int row = blockIdx.z;
    const int section = blockIdx.y;

    const int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
    extern __shared__ float block_vec[];
    
    const int max = x.size(1) - blockDim.x;
    // do first step of the sum in the loading part
    if (i < x.size(1)) {
        if (i < max) {
            const auto sctnTau = tauLo[row][0] + section*tauWidth;
            block_vec[threadIdx.x] = prob_add(x[row][i], x[row][i+blockDim.x], sctnTau, alpha);
        }
        if (i >= max) {
            const auto sctnTau = tauLo[row][0] + section*tauWidth;
            block_vec[threadIdx.x] = prob(x[row][i], sctnTau, alpha);
        }
    }
    if (i >= x.size(1)){
        block_vec[threadIdx.x] = 0.0;
    }
    __syncthreads();
   
    if (blockSize >= 256) {
        if (threadIdx.x < 128) {block_vec[threadIdx.x] += block_vec[threadIdx.x + 128];} __syncthreads(); }
    if (blockSize >= 128) {
        if (threadIdx.x <  64) {block_vec[threadIdx.x] += block_vec[threadIdx.x +  64];} __syncthreads(); }

    if (threadIdx.x < 32) {
        warpReduceSum<blockSize>(block_vec, threadIdx.x);
    }
    
    // let thread 0 for this block write to global memory 
    if (threadIdx.x == 0){
        blockSum[row][section][blockIdx.x] = block_vec[0];
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
template <typename scalar_t, unsigned int blockSize>
__global__ void p_reduction_kernel_lowdim(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tauLo,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pSum,
    scalar_t alpha,
    scalar_t tauWidth
){
    const int row = blockIdx.z;
    const int section = blockIdx.y;

    const int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
    extern __shared__ float row_vec[];
    
    const int max = x.size(1) - blockDim.x;
    // do first step of the sum in the loading part
    if (i < x.size(1)) {
        if (i < max) {
            const auto sctnTau = tauLo[row][0] + section*tauWidth;
            row_vec[threadIdx.x] = prob_add(x[row][i], x[row][i+blockDim.x], sctnTau, alpha);
        }
        if (i >= max) {
            const auto sctnTau = tauLo[row][0] + section*tauWidth;
            row_vec[threadIdx.x] = prob(x[row][i], sctnTau, alpha);
        }
    }
    if (i >= x.size(1)){
        row_vec[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    // if (blockSize >= 1024) {
    //     if (threadIdx.x < 512) {row_vec[threadIdx.x] += row_vec[threadIdx.x + 512];} __syncthreads(); }
    if (blockSize >= 512) {
        if (threadIdx.x < 256) {row_vec[threadIdx.x] += row_vec[threadIdx.x + 256];} __syncthreads(); }
    if (blockSize >= 256) {
        if (threadIdx.x < 128) {row_vec[threadIdx.x] += row_vec[threadIdx.x + 128];} __syncthreads(); }
    if (blockSize >= 128) {
        if (threadIdx.x <  64) {row_vec[threadIdx.x] += row_vec[threadIdx.x +  64];} __syncthreads(); }

    if (threadIdx.x < 32) {
        warpReduceSum<blockSize>(row_vec, threadIdx.x);
    }
    
    // let thread 0 for this block write to global memory 
    if (threadIdx.x == 0){
        pSum[row][section] = row_vec[0];
    }
}
//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------


template <typename scalar_t>
__global__ void sum_reduction_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> blockSum,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pSum
){
    const int row = blockIdx.z;
    const int section = blockIdx.y;

    const int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
    extern __shared__ float sum_vec[];

    const int max = blockSum.size(2) - blockDim.x;
    // do first step of the sum in the loading part
    if (i < blockSum.size(2)) {
        if (i < max) {
            sum_vec[threadIdx.x] = blockSum[row][section][i] + blockSum[row][section][i + blockDim.x];
        }
        if (i >= max) {
            sum_vec[threadIdx.x] = blockSum[row][section][i];
        }
    }
    if (i >= blockSum.size(2)){
        sum_vec[threadIdx.x] = 0.0;
    }
    __syncthreads();    

    // iterate of log base 2 the block dimension
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        // only part of the threads are active
        if (threadIdx.x < s){
            sum_vec[threadIdx.x] += sum_vec[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // thread 0 contains the final sum
    if (threadIdx.x == 0){
        pSum[row][section] = sum_vec[0];
    }


}

template <typename scalar_t>
__global__ void tauLo_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pSum,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tauLo,
    scalar_t tauWidth
){
    const int row = blockIdx.x;
    const int section = threadIdx.x;
    const int maxSctn = pSum.size(1);

    // initialise first section
    extern __shared__ int firstSection[];
    if (pSum[row][section] < 1.0){
        firstSection[section] = section;
    }
    if (pSum[row][section] >= 1.0){
        firstSection[section] = maxSctn;
    }
    // else {
    //     firstSection[section] = pSum.size(1);
    // }
    __syncthreads();

    // iterate of log base 2 the block dimension
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        // only part of the threads are active
        if (threadIdx.x < s){
            if (firstSection[threadIdx.x] > firstSection[threadIdx.x + s]){
                firstSection[threadIdx.x] = firstSection[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    
    // thread 0 contains the final sum
    if (section == 0){
        tauLo[row][0] = tauLo[row][0] + (firstSection[0] - 1) * tauWidth;
    }
}

template <typename scalar_t>
__global__ void p_out_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> tauLo,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pOut,
    scalar_t alpha
){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < x.size(1)){
        pOut[row][col] = prob(x[row][col], tauLo[row][0], alpha);
    }
}

} // namespace

// Standard cuda kernel
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
    

    int threadsP = 512;
    int threadsSum = 32;

    // int threadsP = 64;
    // int threadsSum = 32; // 64
    // if (d > 65536){
    //     threadsP = 256;
    //     threadsSum = 256; // 512
    // }
    // else if (d > 32768){
    //     threadsP = 256;
    //     threadsSum = 128; // 256
    // }
    // else if (d > 16384){
    //     threadsP = 256;
    //     threadsSum = 64; // 128
    // }
    // else if (d > 8192){
    //     threadsP = 128;
    //     threadsSum = 64; // 128
    // }
    // else if (d > 4096){
    //     threadsP = 128;
    //     threadsSum = 32; // 64
    // }
    const int threadsTau = nSections;

    const int blocksdim = (d + threadsP - 1) / threadsP;

    // each thread does double work while loading, so divide threads by two
    const dim3  blocksP((blocksdim + 1) / 2, nSections, bsz);
    const dim3  blocksSum(1, nSections, bsz);
    const dim3  blocksTau(bsz, 1, 1);
    const dim3  blocksPout(blocksdim, bsz, 1);

    auto blockSum = torch::zeros({bsz, nSections, blocksdim}, options);
    auto pSum = torch::zeros({bsz, nSections}, options);

    for(int i = 0; i < nIters; i++){
        tauWidth /= (nSections);

        // kernel for sum over treads in bloock
        switch (threadsP)
        {
        case 1024:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel<scalar_t, 1024><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    blockSum.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 512:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel<scalar_t, 512><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    blockSum.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 256:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel<scalar_t, 256><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    blockSum.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 128:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel<scalar_t, 128><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    blockSum.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 64:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel<scalar_t, 64><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    blockSum.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        }

        // kernel for sum over blocks in grid
        AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
            sum_reduction_kernel<scalar_t><<<blocksSum, threadsSum, threadsSum*4>>>(
                blockSum.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
                );
        }));

        // kernel for updating tauLo
        AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
            tauLo_kernel<scalar_t><<<blocksTau, threadsTau, nSections*4>>>(
                pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                tauWidth
                );
        }));
    }

    auto pOut = torch::zeros_like(x);
    // kernel for sum over threads in blocks
    AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
        p_out_kernel<scalar_t><<<blocksPout, threadsP>>>(
            x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pOut.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha
            );
    
    }));
    return pOut;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Standard cuda kernel for low dim inputs
torch::Tensor entmax_cuda_forward_lowdim(
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
    
    int threadsP = 16; // 32 items
    if (d > 512){
        threadsP = 512; // 1024 items
    }
    else if (d > 256){
        threadsP = 256; // 512 items
    }
    else if (d > 128){
        threadsP = 128; // 256 items
    }
    else if (d > 64){
        threadsP = 64; // 128 items
    }
    else if (d > 32){
        threadsP = 32; // 64 items
    }
    const int threadsTau = nSections;
    const int blocksdim = (d + threadsP - 1) / threadsP;

    // each thread does double work while loading, so divide threads by two
    const dim3  blocksP((blocksdim + 1) / 2, nSections, bsz);
    const dim3  blocksTau(bsz, 1, 1);
    const dim3  blocksPout(blocksdim, bsz, 1);

    auto pSum = torch::zeros({bsz, nSections}, options);

    for(int i = 0; i < nIters; i++){
        tauWidth /= (nSections);

        // kernel for sum over treads in bloock
        switch (threadsP)
        {
        case 512:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel_lowdim<scalar_t, 512><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 256:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel_lowdim<scalar_t, 256><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 128:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel_lowdim<scalar_t, 128><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 64:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel_lowdim<scalar_t, 64><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        case 32:
            AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
                p_reduction_kernel_lowdim<scalar_t, 32><<<blocksP, threadsP, threadsP*4>>>(
                    x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    alpha,
                    tauWidth
                    );
            })); break;
        }

        // kernel for updating tauLo
        AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
            tauLo_kernel<scalar_t><<<blocksTau, threadsTau, nSections*4>>>(
                pSum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                tauWidth
                );
        }));
    }

    auto pOut = torch::zeros_like(x);
    // kernel for sum over threads in blocks
    AT_DISPATCH_FLOATING_TYPES(x.type(), "nsection_forward_cuda", ([&] {
        p_out_kernel<scalar_t><<<blocksPout, threadsP>>>(
            x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            tauLo.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pOut.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha
            );
    
    }));
    return pOut;
}