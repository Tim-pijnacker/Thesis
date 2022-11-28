#include <torch/extension.h>
#include <vector>
#include <iostream>


auto nsection_forward(
    torch::Tensor x,
    float alpha,
    int nIter,
    int nSections
){
    auto options = torch::TensorOptions().device(torch::device_of(x));
    auto shape = torch::_shape_as_tensor(x);
    auto bsz = shape[0].item<int>();
    auto d = shape[1].item<int>();
    auto df = shape[1].item<float>();

    auto max = torch::amax(x, -1, true);
    auto tauLo = max * (alpha - 1) - 1;
    x = x * (alpha - 1);

    df = pow(df, alpha - 1);
    auto tauWidth = (df - 1)/df;
    auto tauFrac = torch::linspace(0,1,nSections, options);

    auto taus = torch::empty({bsz,nSections}, options);
    auto temp = torch::empty({bsz}, options);
    // auto ps = torch::empty({bsz,nSections,d}, torch::dtype(torch::kFloat32).device(torch::device_of(x)));
    auto ps = torch::empty({bsz,nSections,d}, options.dtype(torch::kFloat32));
    auto psd = torch::empty({bsz,nSections,d}, options.dtype(torch::kFloat64));
    auto obj = torch::empty({bsz,nSections}, options);
    auto res = torch::empty({bsz,1}, options.dtype(torch::kInt64));
    // double power = alpha - 1;options
    for(int i = 0; i < nIter; i++){
        torch::add_out(taus,tauLo,tauFrac,tauWidth);
        torch::clamp_min_out(ps, torch::unsqueeze(x, -2) - torch::unsqueeze(taus, -1), 0);
        torch::float_power_out(psd, ps, 1/(alpha - 1));
        torch::sum_out(obj, psd, -1);
        torch::searchsorted_out(res, -obj, -torch::ones({bsz,1}, options));
        torch::index_out(temp, taus, {torch::arange(bsz, options), torch::clamp_min(torch::squeeze(res) - 1, 0)});
        torch::unsqueeze_copy_out(tauLo, temp, -1);
        tauWidth /= nSections;
    }
    auto p = torch::clamp_min(x - tauLo, 0);
    p = torch::float_power(p, 1/(alpha - 1));
    p /= torch::unsqueeze(torch::sum(p,-1), -1); 
    return p;
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
    m.def("forward", &nsection_forward, "Nsection forward");
    m.def("sparsemax_backward", &sparsemax_backward, "Sparsemax backward");
    m.def("entmax_backward", &entmax_backward, "Entmax backward");
}