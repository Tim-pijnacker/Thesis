#include <torch/extension.h>
#include <vector>
#include <iostream>


torch::Tensor prob(
    torch::Tensor Z,
    torch::Tensor tau,
    float alpha    
){
    Z = torch::clamp_min(Z - tau, 0);
    Z = torch::float_power(Z, (1 / (alpha - 1)));
    return Z;
}


torch::Tensor sctn_check(
    torch::Tensor Z,
    torch::Tensor tauLo,
    torch::Tensor tauHi,
    float alpha,
    int dim
){
    auto fLo = prob(Z, tauLo, alpha);
    fLo = torch::sum(fLo,dim) - 1;
    auto fHi = prob(Z, tauHi, alpha);
    fHi = torch::sum(fHi,dim) - 1;
    auto mask = torch::__and__((fLo > 0), (fHi <= 0));
    auto out = torch::unsqueeze(mask, dim);
    return out;
}


torch::Tensor nsection_forward(
    torch::Tensor Z,
    float alpha,
    int dim,
    int nIters,
    int nSections,
    bool ensure_sum_one    
){
    auto shape = torch::_shape_as_tensor(Z);
    auto d = shape[1].item<int>();

    // calculate max 
    auto max = torch::amax(Z, dim, true);
    max = max * (alpha - 1);
    Z = Z * (alpha - 1);

    // calculate starting tau high and low
    auto tauLo = max - 1;
    auto tauHi = max - pow(1 / (double) d, alpha - 1);

    auto sctnRng = tauHi - tauLo;
    auto sctnTauLo = torch::zeros_like(tauLo);
    auto sctnTauHi = torch::zeros_like(tauLo);
    // input_mat_cuda = torch:zeros_like(...)
    for(int i = 0; i < nIters; i++){
        sctnRng /= nSections;
        for(int j = 1; j < nSections+1; j++){
            torch::add_outf(tauLo, sctnRng, j - 1, sctnTauLo);
            torch::add_outf(tauLo, sctnRng, j, sctnTauHi);
            auto mask = sctn_check(Z, sctnTauLo, sctnTauHi, alpha, dim);
            torch::where_out(tauLo, mask, sctnTauLo, tauLo);
        } 
    }
    torch::Tensor p = prob(Z, tauLo, alpha);
    if(ensure_sum_one){
        p /= torch::unsqueeze(torch::sum(p,dim), dim);
    }
    return p;
}


torch::Tensor sparsemax_backward(
    torch::Tensor Y,
    torch::Tensor dY,
    int dim
){
    auto gppr = torch::where(Y > 0, 1.0, 0.0);
    auto dX = dY * gppr;
    auto q =  torch::sum(dX,dim) / torch::sum(gppr,dim);
    q = torch::unsqueeze(q, dim);
    dX -= q * gppr;
    return dX;
}


torch::Tensor entmax_backward(
    torch::Tensor Y,
    torch::Tensor dY,
    torch::Tensor alpha,
    int dim
){
    auto gppr = torch::where(Y > 0, torch::float_power(Y, (2 - alpha)), torch::zeros({1}));
    auto dX = dY * gppr;
    auto q =  torch::sum(dX,dim) / torch::sum(gppr,dim);
    q = torch::unsqueeze(q, dim);
    dX -= q * gppr;
    return dX;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &nsection_forward, "Nsection forward");
    m.def("sparsemax_backward", &sparsemax_backward, "Sparsemax backward");
    m.def("entmax_backward", &entmax_backward, "Entmax backward");
}