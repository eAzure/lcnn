/**
 * op.cc
 * [by lgx 2023-02-17]
*/

#include "ir/op/op.h"
namespace lcnn {

InferStatus Op::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    std::cout << this->_name << " op not implement yet!" << std::endl;
    return InferStatus::kInferUnknown;
}

} // namespace lcnn