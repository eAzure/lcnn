/**
 * tensor.cc
 * [by lgx 2023-02-10]
*/

#include "spdlog/spdlog.h"
#include "tensor/tensor.h"

namespace lcnn {

// temp for test
void Tensor<float>::test() {
    spdlog::info("Welcome to spdlog!");
    spdlog::info("tensor's rows is {}", _data.dimensions()[1]);
}

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    _data = Eigen::Tensor<float, 3, Eigen::RowMajor>(channels, rows, cols);
}

} // namespace lcnn
