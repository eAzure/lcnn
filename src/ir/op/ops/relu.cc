/**
 * relu.cc
 * [by lgx 2023-02-17]
*/

#include "ir/op/ops/relu.h"
#include "ir/op/op_register.h"

namespace lcnn {

InferStatus ReluOp::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    // Check
    std::cout << "op forward: " << this->op_name() << std::endl;
    if (inputs.empty()) {
        std::cout << "The input feature of relu op is empty." << std::endl;
        return InferStatus::kInferFailedInputEmpty;
    }
    if (inputs.size() != outputs.size()) {
        std::cout << "The input size is not equal to the output." << std::endl;
        return InferStatus::kInferFailedInputOutputSizeNotEqual;
    }

    const uint32_t batch_size = inputs.size();
    for (uint32_t i=0;i<batch_size;i++) {
        const sptrFloatTensor &input_data = inputs.at(i);
        sptrFloatTensor &output_data = outputs.at(i);
        if (input_data == nullptr || input_data->empty()) {
            std::cout << "The input feature of relu op is empty." << std::endl;
            return InferStatus::kInferFailedInputEmpty;
        }
        if (output_data != nullptr && !output_data->empty()) {
            if (input_data->shapes() != output_data->shapes()) {
                std::cout << "The input size is not equal to the output." << std::endl;
                return InferStatus::kInferFailedInputOutputSizeNotEqual;
            }
        }

        // Compute
        if (output_data == nullptr || output_data->empty()) {
            output_data = std::make_shared<Tensor<float>>(input_data->shapes());
            outputs.at(i) = output_data;
        }
        output_data->set_data(input_data->data());
        output_data->Transform([](const float val)->float{
            return val > 0. ? val : 0.;
        });
    }

    return InferStatus::kInferSuccess;
}

// op Creator
ParseParameterAttrStatus ReluOp::GetInstance(const std::shared_ptr<Operator> &con_operator,
                                             std::shared_ptr<Op> &relu_op) {
    if (con_operator == nullptr) {
        std::cout << "[Error] Relu operator is nullptr";
        return ParseParameterAttrStatus::kParameterMissingUnknown;
    }
    relu_op = std::make_shared<ReluOp>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

// 前面定义的是op_type，后面定义的是OpCreator方法，由此便将op_type与op对应的创建方法绑定到了一起
OpRegistererWrapper kReluGetInstance("nn.Relu", ReluOp::GetInstance);


} // namespace lcnn
