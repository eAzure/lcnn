/**
 * tensor.h
 * [by lgx 2023-02-10]
*/

#ifndef _LCNN_TENSOR_H
#define _LCNN_TENSOR_H

#include "armadillo"
#include <memory>
#include <vector>
#include <iostream>
#include <assert.h>

namespace lcnn {

template<typename T>
class Tensor {};

template<>
class Tensor<float> {
public:
    void test(); // temp for test

    // construct
    explicit Tensor() = default;
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    explicit Tensor(const std::vector<uint32_t> &shapes);
    Tensor(const Tensor &tensor); // 拷贝构造
    Tensor(Tensor &&tensor) noexcept; // 移动构造
    Tensor<float> &operator=(const Tensor &tensor); // 拷贝赋值运算
    Tensor<float> &operator=(Tensor &&tensor) noexcept; // 移动赋值运算符

    //
    uint32_t rows() const; // 返回行数
    uint32_t cols() const; // 返回列数
    uint32_t channels() const; // 返回通道数
    uint32_t size() const; // 返回张量中元素数量

    //
    arma::fcube &data(); // 返回张量中的数据
    const arma::fcube &data() const;
    const float* raw_ptr() const; // 返回数据的原始指针

    //
    void set_data(const arma::fcube &data); // 设置张量中的具体数据
private:
    std::vector<uint32_t> _raw_shapes; // 实际尺寸大小
    arma::fcube _data;
};

} // namespace lcnn

#endif // _LCNN_TENSOR_H