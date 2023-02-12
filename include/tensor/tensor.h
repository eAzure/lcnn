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
    std::vector<uint32_t> shapes() const; // 返回张量尺寸
    const std::vector<uint32_t> &raw_shapes() const; // 返回张量实际尺寸大小

    //
    arma::fcube &data(); // 返回张量中的数据
    const arma::fcube &data() const;
    const float* raw_ptr() const; // 返回数据的原始指针

    //
    void set_data(const arma::fcube &data); // 设置张量中的具体数据

    // 
    bool empty() const; // 判断张量是否非空

    // 一些访存操作
    // 列主序
    float index(uint32_t offset) const; // 返回张量中offset位置的元素
    float &index(uint32_t offset);
    const arma::fmat &at(uint32_t channel) const;// 返回张量中第channel通道中的数据
    arma::fmat &at(uint32_t channel);
    // 行主序
    float at(uint32_t channel, uint32_t row, uint32_t col) const; // 返回张量中特定位置的元素
    float &at(uint32_t channel, uint32_t row, uint32_t col);

    // 数据填充
    void Fill(float value); // 使用value去填充张量
    void Fill(const std::vector<float> &values); // 使用values去填充张量
    void Ones(); // 使用常量1初始化张量
    void Rand(); // 使用随机值初始化张量

    // 打印张量
    void Show();

    // 张量变换
    void Padding(const std::vector<uint32_t> &pads, float padding_value); // 填充
    void Flatten(); // 展开张量
    void Transform(const std::function<float(float)> &filter); // 对张量中的每一个元素进行过滤

    // shape变换
    // reshape: 列主序
    /**
     * 1 2
     * 3 4
     * -> 1 3 2 4
    */
    void ReRawShape(const std::vector<uint32_t> &shapes);
    // reshape: 行主序
    /**
     * 1 2
     * 3 4
     * -> 1 2 3 4
    */
    void ReRawView(const std::vector<uint32_t> &shapes);

    //
    std::shared_ptr<Tensor> Clone(); // clone该Tensor，返回指向该Tensor的指针

private:
    std::vector<uint32_t> _raw_shapes; // 实际尺寸大小
    arma::fcube _data;
};

} // namespace lcnn

#endif // _LCNN_TENSOR_H