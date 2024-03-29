#include <torch/extension.h>
#include <vector>

// cuda declarations
torch::Tensor entmax_cuda_forward(
    torch::Tensor Z,
    float alpha,
    int nIters,
    int nSections
);

torch::Tensor entmax_cuda_forward_lowdim(
    torch::Tensor Z,
    float alpha,
    int nIters,
    int nSections
);

torch::Tensor sparsemax_cuda_forward(
    torch::Tensor Z,
    int nIters,
    int nSections
);

torch::Tensor sparsemax_cuda_forward_lowdim(
    torch::Tensor Z,
    int nIters,
    int nSections
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor nsection_forward(
    torch::Tensor Z,
    float alpha,
    int nIters,
    int nSections
){
    CHECK_INPUT(Z);

    auto shape = torch::_shape_as_tensor(Z);
    auto d = shape[1].item<int>();

    if (d <= 1024) {
        return entmax_cuda_forward_lowdim(Z, alpha, nIters, nSections);
    }
    return entmax_cuda_forward(Z, alpha, nIters, nSections);
}

torch::Tensor sparsemax_nsection_forward(
    torch::Tensor Z,
    int nIters,
    int nSections
){
    CHECK_INPUT(Z);

    auto shape = torch::_shape_as_tensor(Z);
    auto d = shape[1].item<int>();

    if (d <= 1024) {
        return sparsemax_cuda_forward_lowdim(Z, nIters, nSections);
    }
    return sparsemax_cuda_forward(Z, nIters, nSections);
}


torch::Tensor sparsemax_backward(
    torch::Tensor Y,
    torch::Tensor dY
){
    auto gppr = torch::where(Y > 0, 1.0, 0.0);
    auto dX = dY * gppr;
    auto q =  torch::sum(dX,-1) / torch::sum(gppr, -1);
    q = torch::unsqueeze(q, -1);
    dX -= q * gppr;
    return dX;
}


torch::Tensor entmax_backward(
    torch::Tensor Y,
    torch::Tensor dY,
    float alpha
){
    auto gppr = torch::where(Y > 0, torch::float_power(Y, (2 - alpha)), torch::zeros({1}));
    auto dX = dY * gppr;
    auto q =  torch::sum(dX,-1) / torch::sum(gppr,-1);
    q = torch::unsqueeze(q, -1);
    dX -= q * gppr;
    return dX;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &nsection_forward, "nsection forward");
    m.def("sparsemax_forward", &sparsemax_nsection_forward, "nsection forward1");
    m.def("sparsemax_backward", &sparsemax_backward, "Sparsemax backward");
    m.def("entmax_backward", &entmax_backward, "Entmax backward");
}