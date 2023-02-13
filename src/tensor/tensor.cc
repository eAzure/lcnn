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
// 返回张量尺寸
std::vector<uint32_t> Tensor<float>::shapes() const {
    assert(!this->_data.empty());
    return {this->channels(), this->rows(), this->cols()};
}
// 返回张量实际尺寸大小
const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
    assert(!this->_raw_shapes.empty());
    return this->_raw_shapes;
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

// 判断张量是否非空
bool Tensor<float>::empty() const {
    return this->_data.empty();
}

// 一些访存操作
// 获得offset位置的数据
float Tensor<float>::index(uint32_t offset) const {
    assert(offset < this->_data.size());
    return this->_data.at(offset);
}
float& Tensor<float>::index(uint32_t offset) {
    assert(offset < this->_data.size());
    return this->_data.at(offset);
}
// 获取第channel的通道的数据
arma::fmat& Tensor<float>::at(uint32_t channel) {
    assert(channel < this->channels());
    return this->_data.slice(channel);
}
const arma::fmat& Tensor<float>::at(uint32_t channel) const {
    assert(channel < this->channels());
    return this->_data.slice(channel);
}
// 获取坐标是(channel, row, col)的数据
float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
    assert(row < this->rows());
    assert(col < this->cols());
    assert(channel < this->channels());
    return this->_data.at(row, col, channel);
}
float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
    assert(row < this->rows());
    assert(col < this->cols());
    assert(channel < this->channels());
    return this->_data.at(row, col, channel);
}

// 填充数据
// 以value值填充数据
void Tensor<float>::Fill(float value) {
    assert(!this->_data.empty());
    this->_data.fill(value);
}
// 以vector<float>value去填充数据
void Tensor<float>::Fill(const std::vector<float>& values) {
    assert(!this->_data.empty());
    assert(values.size() == this->_data.size());

    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->_data.n_slices;

    for (uint32_t i=0;i<channels;i++) {
        auto& channel_data = this->_data.slice(i);
        const arma::fmat& channel_data_t = 
            arma::fmat(values.data() + i*planes, this->cols(), this->rows());
        channel_data = channel_data_t.t();
    }
}
// 以常量1初始化张量
void Tensor<float>::Ones() {
    assert(!this->_data.empty());
    this->_data.fill(1.);
}
// 随机初始化张量
void Tensor<float>::Rand() {
    assert(!this->_data.empty());
    this->_data.randn();
}

// 打印张量
void Tensor<float>::Show() {
    for (uint32_t i=0; i<this->channels();i++) {
        std::cout << "Channel " << i << ":" << std::endl;
        std::cout << this->_data.slice(i);
    }
}

/********* 张量变换 *********/
// 张量填充
void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
    assert(!this->_data.empty());
    assert(pads.size() == 4);
    uint32_t pad_rows1 = pads.at(0); // top
    uint32_t pad_rows2 = pads.at(1); // bottom
    uint32_t pad_cols1 = pads.at(2); // left
    uint32_t pad_cols2 = pads.at(3); // right

    arma::fcube padding_data(this->_data.n_rows + pad_rows1 + pad_rows2,
                             this->_data.n_cols + pad_cols1 + pad_cols2,
                             this->_data.n_slices);
    padding_data.fill(padding_value);
    // subcube(first_row, first_col, first_slice, second_row, second_col, second_slice)
    padding_data.subcube(pad_rows1, pad_cols1, 0,
                         padding_data.n_rows - pad_rows2 - 1,
                         padding_data.n_cols - pad_cols2 - 1,
                         padding_data.n_slices - 1) = this->_data;
    this->_data = std::move(padding_data);
}
// 张量展开
void Tensor<float>::Flatten() {
    assert(!this->_data.empty());
    const uint32_t size = this->_data.size();
    this->ReRawShape({size});
}
// 张量筛选
void Tensor<float>::Transform(const std::function<float(float)> &filter) {
    assert(!this->_data.empty());
    uint32_t channels = this->channels();
    for (uint32_t c=0;c<channels;c++) {
        this->_data.slice(c).transform(filter);
    }
}

/********* shape变换 *********/
// reshape 列主序
void Tensor<float>::ReRawShape(const std::vector<uint32_t> &shapes) {
    assert(!this->_data.empty());
    assert(!shapes.empty());
    assert(shapes.size() <= 3);
    const uint32_t origin_size = this->size();
    uint32_t current_size = 1;
    for (uint32_t s : shapes) {
        current_size *= s;
    }
    assert(current_size == origin_size);

    if (shapes.size() == 3) {
        this->_data.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
        this->_raw_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
        this->_data.reshape(shapes.at(0), shapes.at(1), 1);
        this->_raw_shapes = {shapes.at(0), shapes.at(1)};
    } else {
        this->_data.reshape(shapes.at(0), 1, 1);
        this->_raw_shapes = {shapes.at(0)};
    }
}
// reshape 行主序
void Tensor<float>::ReRawView(const std::vector<uint32_t> &shapes) {
    assert(!this->_data.empty());
    assert(!shapes.empty());
    assert(shapes.size() <= 3);
    const uint32_t origin_size = this->size();
    uint32_t current_size = 1;
    for (uint32_t s : shapes) {
        current_size *= s;
    }
    assert(current_size == origin_size);

    std::vector<uint32_t> target_shapes;
    if (shapes.size() == 3) {
        target_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
        this->_raw_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
        target_shapes = {1, shapes.at(0), shapes.at(1)};
        this->_raw_shapes = {shapes.at(0), shapes.at(1)};
    } else {
        target_shapes = {1, shapes.at(0), 1};
        this->_raw_shapes = {shapes.at(0)};
    }

    const uint32_t target_channels = target_shapes.at(0);
    const uint32_t target_rows = target_shapes.at(1);
    const uint32_t target_cols = target_shapes.at(2);
    arma::fcube new_data(target_rows, target_cols, target_channels);

    const uint32_t plane_size = target_rows * target_cols;
    for (uint32_t c=0;c<this->_data.n_slices;c++) {
        const arma::fmat& channel = this->_data.slice(c);
        for (uint32_t w=0;w<this->_data.n_cols;w++) {
            const float* col_ptr = channel.colptr(w);
            for (uint32_t h=0;h<this->_data.n_rows;h++) {
                const uint32_t pos_index = c * this->_data.n_rows * this->_data.n_cols +
                                           h * this->_data.n_cols + w;
                const uint32_t new_c = pos_index / plane_size;
                const uint32_t new_h = (pos_index % plane_size) / target_cols;
                const uint32_t new_w = (pos_index % plane_size) % target_cols;
                new_data.at(new_h, new_w, new_c) = *(col_ptr + h);
            }
        }
    }
    this->_data = new_data;
}

// Clone
std::shared_ptr<Tensor<float>> Tensor<float>::Clone() {
    return std::make_shared<Tensor>(*this);
}

/********* 处理Tensor的相关库 *********/
// 创建Tensor
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
    return std::make_shared<Tensor<float>>(channels, rows, cols);
}
std::shared_ptr<Tensor<float>> TensorCreate(const std::vector<uint32_t>& shapes) {
    assert(shapes.size() == 3);
    return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}
// 判断两tensor 数值是否相同
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& tensor1,
                  const std::shared_ptr<Tensor<float>>& tensor2) {
    assert(tensor1 != nullptr);
    assert(tensor2 != nullptr);
    if (tensor1->shapes() != tensor2->shapes()) {
        return false;
    }
    bool is_same = arma::approx_equal(tensor1->data(), tensor2->data(), "absdiff", 1e-5);
    return is_same;
}
// 实现广播，将两Tensor按照广播机制调整到相同尺寸
// TODO: 更改现有规则
/**
 * 现有规则：
 * channel相同，然后有一方的row和col均为1
*/
std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>> 
    TensorBroadCast(const std::shared_ptr<Tensor<float>>& tensor1,
                    const std::shared_ptr<Tensor<float>>& tensor2) {
    assert(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        return {tensor1, tensor2};
    } else {
        assert(tensor1->channels() == tensor2->channels());
        if (tensor1->rows() == 1 && tensor1->rows() == 1) {
            std::shared_ptr<Tensor<float>> new_tensor1 = 
                TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
            for (uint32_t c=0;c<tensor1->channels();c++) {
                new_tensor1->at(c).fill(tensor1->index(c));
            }
            return {new_tensor1, tensor2};
        } else if (tensor2->rows() == 1 && tensor2->cols() == 1) {
            std::shared_ptr<Tensor<float>> new_tensor2 = 
                TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
            for (uint32_t c=0;c<tensor2->channels();c++) {
                new_tensor2->at(c).fill(tensor2->index(c));
            }
            return {tensor1, new_tensor2};
        } else {
            std::cout << "----现有广播规则不支持！" << std::endl;
            return {nullptr, nullptr};
        }
    }
}
// Tensor Elementwise 加
std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
    assert(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor<float>> output_tensor =
            TensorCreate(tensor1->shapes());
        output_tensor->set_data(tensor1->data() + tensor2->data());
        return output_tensor;
    } else {
        // need broadcast
        const auto& tuple_tensor = TensorBroadCast(tensor1, tensor2);
        const auto& new_tensor_1 = std::get<0>(tuple_tensor);
        const auto& new_tensor_2 = std::get<1>(tuple_tensor);
        if (new_tensor_1 == nullptr || new_tensor_2 == nullptr) return nullptr;
        std::shared_ptr<Tensor<float>> output_tensor = 
            TensorCreate(new_tensor_1->shapes());
        output_tensor->set_data(new_tensor_1->data() + new_tensor_2->data());
        return output_tensor;
    }
}
void TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output) {
    assert(tensor1 != nullptr && tensor2 != nullptr && output != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        assert(tensor1->shapes() == output->shapes());
        output->set_data(tensor1->data() + tensor2->data());
    } else {
        // need broadcast
        const auto& tuple_tensor = TensorBroadCast(tensor1, tensor2);
        const auto& new_tensor_1 = std::get<0>(tuple_tensor);
        const auto& new_tensor_2 = std::get<1>(tuple_tensor);
        assert(new_tensor_1->shapes() == output->shapes());
        output->set_data(new_tensor_1->data() + new_tensor_2->data());
    }
}
// Tensor Elementwise 相乘
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {

    assert(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor<float>> output_tensor =
            TensorCreate(tensor1->shapes());
        output_tensor->set_data(tensor1->data() % tensor2->data());
        return output_tensor;
    } else {
        // need broadcast
        const auto& tuple_tensor = TensorBroadCast(tensor1, tensor2);
        const auto& new_tensor_1 = std::get<0>(tuple_tensor);
        const auto& new_tensor_2 = std::get<1>(tuple_tensor);
        if (new_tensor_1 == nullptr || new_tensor_2 == nullptr) return nullptr;
        std::shared_ptr<Tensor<float>> output_tensor = 
            TensorCreate(new_tensor_1->shapes());
        output_tensor->set_data(new_tensor_1->data() % new_tensor_2->data());
        return output_tensor;
    }
}
void TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output) {

    assert(tensor1 != nullptr && tensor2 != nullptr && output != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        assert(tensor1->shapes() == output->shapes());
        output->set_data(tensor1->data() % tensor2->data());
    } else {
        // need broadcast
        const auto& tuple_tensor = TensorBroadCast(tensor1, tensor2);
        const auto& new_tensor_1 = std::get<0>(tuple_tensor);
        const auto& new_tensor_2 = std::get<1>(tuple_tensor);
        assert(new_tensor_1->shapes() == output->shapes());
        output->set_data(new_tensor_1->data() % new_tensor_2->data());
    }
}

} // namespace lcnn
