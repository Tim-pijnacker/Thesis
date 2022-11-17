#include <torch/extension.h>
#include <vector>
#include <iostream>


torch::Tensor prob(
    torch::Tensor Z,
    torch::Tensor tau,
    torch::Tensor alpha    
){
    torch::Scalar min = 0;
    Z = torch::clamp_min(Z - tau, min);
    torch::Tensor power = 1 / (alpha - 1);
    Z = torch::float_power(Z, power);
    return Z;
}


torch::Tensor sctn_check(
    torch::Tensor Z,
    torch::Tensor tauLo,
    torch::Tensor tauHi,
    torch::Tensor alpha,
    int dim
){
    torch::Tensor fLo = prob(Z, tauLo, alpha);
    fLo = torch::sum(fLo,dim) - 1;
    torch::Tensor fHi = prob(Z, tauHi, alpha);
    fHi = torch::sum(fHi,dim) - 1;
    torch::Tensor mask = torch::__and__((fLo > 0), (fHi <= 0));
    torch::Tensor out = torch::unsqueeze(mask, dim);
    return out;
}


torch::Tensor nsection_forward(
    torch::Tensor Z,
    torch::Tensor tau,
    float alphaFlt,
    int nIters
    int dim
){
    // create alpha tensor
    torch::Tensor shape = torch::_shape_as_tensor(Z);
    int d = shape[1].item<int>();
    shape[dim] = 1;
    torch::Tensor alpha = torch::ones({shape[0].item<int>(), shape[1].item<int>()}) + (alphaFlt - 1);

    // calculate max 
    torch::Tensor max = torch::amax(Z, dim, true);
    max = max * (alphaFlt - 1);
    Z = Z * (alphaFlt - 1);

    // calculate starting tau high and low
    torch::Tensor tauLo = max - 1;
    float factor = 1 / (float) d; 
    torch::Tensor power = alpha - 1;
    torch::Tensor tauHi = max - torch::float_power(factor, power);
    
    torch::Tensor sctnRng = tauHi - tauLo;
    for(int i = 0, i < nIter, i++){

    }
    return sctnRng;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &nsection_forward, "Nsection forward");
    m.def("sctn_check", &sctn_check, "section check");
}