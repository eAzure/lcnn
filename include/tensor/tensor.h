/**
 * tensor.h
 * [by lgx 2023-02-10]
*/

#ifndef _LCNN_TENSOR_H
#define _LCNN_TENSOR_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"

namespace lcnn {

template<typename T>
class Tensor {};

template<>
class Tensor<float> {
public:
    void test(); // temp for test
    explicit Tensor() = default;
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
private:
    Eigen::Tensor<float, 3, Eigen::RowMajor> _data;
};

} // namespace lcnn

#endif // _LCNN_TENSOR_H