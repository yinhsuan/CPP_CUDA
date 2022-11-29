// #include "include/utils.h"
#include <torch/extension.h>

 torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor points
){
    return feats;
    // CHECK_INPUT(feats);
    // CHECK_INPUT(points);

    // return trilinear_fw_cu(feats, points);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    // define python要叫的名稱 & C++的程式是什麼
    m.def("trilinear_interpolation", &trilinear_interpolation);
}