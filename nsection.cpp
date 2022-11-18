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
    torch::Tensor alpha,
    float alphaVal,
    int dim,
    int nIters,
    int nSections,
    bool ensure_sum_one    
){
    torch::Tensor shape = torch::_shape_as_tensor(Z);
    int d = shape[1].item<int>();

    // calculate max 
    torch::Tensor max = torch::amax(Z, dim, true);
    max = max * (alphaVal - 1);
    Z = Z * (alphaVal - 1);

    // calculate starting tau high and low
    torch::Tensor tauLo = max - 1;
    double factor = 1 / (double) d; 
    torch::Tensor power = alpha - 1;
    torch::Tensor tauHi = max - torch::float_power(factor, power);

    torch::Tensor sctnRng = tauHi - tauLo;
    for(int i = 0; i < nIters; i++){
        sctnRng /= nSections;
        for(int j = 1; j < nSections+1; j++){
            torch::Tensor sctnTauLo = tauLo + (j-1)*sctnRng;
            torch::Tensor sctnTauHi = tauLo + j*sctnRng;
            torch::Tensor mask = sctn_check(Z, sctnTauLo, sctnTauHi, alpha, dim);
            tauLo = torch::where(mask, sctnTauLo, tauLo);
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
    torch::Tensor gppr = torch::where(Y > 0, 1.0, 0.0);
    torch::Tensor dX = dY * gppr;
    torch::Tensor q =  torch::sum(dX,dim) / torch::sum(gppr,dim);
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
    torch::Tensor gppr = torch::where(Y > 0, torch::float_power(Y, (2 - alpha)), torch::zeros({1}));
    torch::Tensor dX = dY * gppr;
    torch::Tensor q =  torch::sum(dX,dim) / torch::sum(gppr,dim);
    q = torch::unsqueeze(q, dim);
    dX -= q * gppr;
    return dX;
}

torch::Tensor alpha_gradient(
    torch::Tensor Y,
    torch::Tensor dY,
    torch::Tensor alpha,
    int dim
){
    torch::Tensor S = torch::where(Y > 0, Y * torch::log(Y), torch::zeros({1}));
    torch::Tensor ent = torch::unsqueeze(torch::sum(S,dim), dim);
    torch::Tensor gppr = torch::where(Y > 0, torch::float_power(Y, (2 - alpha)), torch::zeros({1}));
    torch::Tensor Y_skewed = gppr / torch::unsqueeze(torch::sum(gppr,dim), dim);
    torch::Tensor d_alpha =  dY * (Y - Y_skewed) / torch::float_power(alpha-1, 2);
    d_alpha -= dY * (S - Y_skewed * ent) / (alpha - 1);
    d_alpha = torch::unsqueeze(torch::sum(d_alpha,dim), dim);
    return d_alpha;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &nsection_forward, "Nsection forward");
    m.def("sparsemax_backward", &sparsemax_backward, "Sparsemax backward");
    m.def("entmax_backward", &entmax_backward, "Entmax backward");
    m.def("alpha_gradient", &alpha_gradient, "Alpha gradient");
}