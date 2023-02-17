/**
 * op.h
 * 主要目的用于封装具体的算数逻辑
 * [by lgx 2023-02-17]
*/

#ifndef _LCNN_IR_OP_H
#define _LCNN_IR_OP_H

#include <string>
#include "tensor/tensor.h"
#include "utils/status_code.h"

namespace lcnn {

class Op {
public:
    explicit Op(std::string name):_name(std::move(name)) {}
    virtual ~Op() = default;
    // 前向计算
    virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs);
    // 返回Op _name
    virtual const std::string &op_name() const {
        return this->_name;
    }
private:
    std::string _name; // Op名称
};

} // namespace lcnn

#endif // _LCNN_IR_OP_H