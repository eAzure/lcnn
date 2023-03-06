/**
 * operand.h
 * 操作数
 * [by lgx 2023-02-25]
*/

#ifndef _LCNN_IR_OPERAND_H
#define _LCNN_IR_OPERAND_H

#include "tensor/tensor.h"
#include "utils/data_type.h"

namespace lcnn {

struct Operand
{
    // 操作数名称
    std::string name;
    // 操作数形状
    std::vector<int32_t> shape;
    // 对应的Tensor数据
    std::vector<std::shared_ptr<Tensor<float>>> datas;
    // 操作数类型
    OperandDataType type = OperandDataType::kTypeUnknown;
};

} // namespace lcnn

#endif // _LCNN_IR_OPERAND_H