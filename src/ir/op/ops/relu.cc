/**
 * relu.cc
 * [by lgx 2023-02-17]
*/

#include "ir/op/ops/relu.h"
namespace lcnn {

InferStatus ReluOp::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    std::cout << "op forward: " << this->op_name() << std::endl;
    return InferStatus::kInferSuccess;
}

} // namespace lcnn
