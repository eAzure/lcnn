/**
 * operator.h
 * 计算图中的算子节点——Operator
 * [by lgx 2023-02-22]
*/

#ifndef _LCNN_IR_OPERATOR_H
#define _LCNN_IR_OPERATOR_H

#include "ir/op/op.h"

namespace lcnn {

struct Operator
{
    std::string name; // 计算节点名称 如conv1
    std::string type; // 计算节点对应的类型 如nn.Conv2d
    std::shared_ptr<Op> op; // 对应的具体计算op
};


} // namespace lcnn

#endif // _LCNN_IR_OPERATOR_H
