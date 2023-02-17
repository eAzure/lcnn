/**
 * relu.h
 * relu op
 * [by lgx 2023-02-17]
*/

#ifndef _LCNN_IR_OPS_RELU_H
#define _LCNN_IR_OPS_RELU_H

#include "ir/op/op.h"

namespace lcnn {

class ReluOp:public Op {
public:
    ReluOp():Op("Relu") {}
    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
};

} // namespace lcnn

#endif // _LCNN_IR_OPS_RELU_H
