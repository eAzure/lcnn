/**
 * op.cc
 * [by lgx 2023-02-17]
*/

// #include "ir/op/op.h"
#include "ir/operator.h"
namespace lcnn {

// 绑定operator
void Op::set_operator(const std::shared_ptr<Operator>& con_operator) {
    this->_operator = con_operator;
}

InferStatus Op::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    std::cout << this->_name << " op not implement yet!" << std::endl;
    return InferStatus::kInferUnknown;
}

// 用于计算图遍历节点时候执行，将数据准备过程进行封装
InferStatus Op::Forward() {
    if (this->_operator.expired()) {
        std::cout << "[Error] The operator is expired or nullptr!" << std::endl;
        return InferStatus::kInferFailed;
    }
    // 如果当前 weak_ptr 已经过期，则该函数会返回一个空的 shared_ptr 指针；反之，该函数返回一个和当前 weak_ptr 指向相同的 shared_ptr 指针。
    const auto& curr_operator = this->_operator.lock();
    // 获得该节点的操作数
    const std::vector<std::shared_ptr<Operand>>& input_operands = curr_operator->input_operands_seq;
    // 准备数据输入
    std::vector<std::shared_ptr<Tensor<float>>> input_datas;
    for (const auto& input_operand : input_operands) {
        for (const auto& input_data : input_operand->datas) {
            input_datas.push_back(input_data);
        }
    }
    if (input_datas.empty()) {
        std::cout << "Op input data is empty!" << std::endl;
        return InferStatus::kInferFailed;
    }
    if (curr_operator->output_operand == nullptr || curr_operator->output_operand->datas.empty()) {
        std::cout << "Op output data is empty!" << std::endl;
        return InferStatus::kInferFailed;
    }
    // 实际执行
    InferStatus status = curr_operator->op->Forward(input_datas, curr_operator->output_operand->datas);
    return status;
}

} // namespace lcnn