#include <torch/extension.h>
#include <vector>
#include <iostream>


// def sctn_check(cls, X, tau_lo: float, tau_hi: float, alpha: float, dim):
//     f_lo = cls.p(X, tau_lo, alpha)
//     f_lo = f_lo.sum(dim) - 1
//     f_hi = cls.p(X, tau_hi, alpha)
//     f_hi = f_hi.sum(dim) - 1
//     out = ((f_lo > 0) & (f_hi <= 0)).unsqueeze(dim)
//     return out


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
    torch::Tensor tau_lo,
    torch::Tensor tau_hi,
    torch::Tensor alpha,
    torch::Scalar dim
){}


torch::Tensor nsection_forward(
    torch::Tensor Z,
    torch::Tensor tau,
    torch::Tensor alpha 
){
    torch::Tensor output = prob(Z, tau, alpha);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nsection_forward, "Nsection forward");
}