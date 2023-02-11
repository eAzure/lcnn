/**
 * tensor.cc
 * [by lgx 2023-02-10]
*/

#include "tensor/tensor.h"

namespace lcnn {

// temp for test
void Tensor<float>::test() {
    std::cout << "tensor's rows is " << _data.n_rows << std::endl;
}

// construct
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    this->_data = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        this->_raw_shapes = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        this->_raw_shapes = std::vector<uint32_t>{rows, cols};
    } else {
        this->_raw_shapes = std::vector<uint32_t>{channels, rows, cols};
    }
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
    assert(shapes.size() == 3);
    uint32_t channels = shapes.at(0);
    uint32_t rows = shapes.at(1);
    uint32_t cols = shapes.at(2);

    this->_data = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        this->_raw_shapes = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        this->_raw_shapes = std::vector<uint32_t>{rows, cols};
    } else {
        this->_raw_shapes = std::vector<uint32_t>{channels, rows, cols};
    }
}
// 拷贝构造
Tensor<float>::Tensor(const Tensor& tensor) {
    if (this != &tensor) {
        std::cout << "---------- 拷贝构造 ----------" << std::endl;
        this->_data = tensor._data;
        this->_raw_shapes = tensor._raw_shapes;
    }
}
// 移动构造
Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        std::cout << "---------- 移动构造 ----------" << std::endl;
        this->_data = std::move(tensor._data); // 实现完美转发
        this->_raw_shapes = tensor._raw_shapes;
    }
}
// 拷贝赋值运算
Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
    if (this != &tensor) {
        std::cout << "---------- 拷贝赋值 ----------" << std::endl;
        this->_data = tensor._data;
        this->_raw_shapes = tensor._raw_shapes;
    }
    return *this;
}
// 移动赋值运算
Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        std::cout << "---------- 移动赋值 ----------" << std::endl;
        this->_data = std::move(tensor._data);
        this->_raw_shapes = _raw_shapes;
    }
    return *this;
}

// 返回行数
uint32_t Tensor<float>::rows() const {
    assert(!this->_data.empty());
    return this->_data.n_rows;
}
// 返回列数
uint32_t Tensor<float>::cols() const {
    assert(!this->_data.empty());
    return this->_data.n_cols;
}
// 返回通道数
uint32_t Tensor<float>::channels() const {
    assert(!this->_data.empty());
    return this->_data.n_slices;
}
// 返回size
uint32_t Tensor<float>::size() const {
    assert(!this->_data.empty());
    return this->_data.size();
}

// 返回张量中的数据
arma::fcube& Tensor<float>::data() {
    return this->_data;
}
const arma::fcube& Tensor<float>::data() const {
    return this->_data;
}
// 返回数据的原始指针
const float* Tensor<float>::raw_ptr() const {
    return this->_data.memptr();
}

// 设置张量中的具体操作
void Tensor<float>::set_data(const arma::fcube &data) {
    assert(this->_data.n_rows == data.n_rows);
    assert(this->_data.n_cols == data.n_cols);
    assert(this->_data.n_slices == data.n_slices);
    this->_data = data;
}

} // namespace lcnn
