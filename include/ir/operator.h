/**
 * operator.h
 * 计算图中的算子节点——Operator
 * [by lgx 2023-02-22]
*/

#ifndef _LCNN_IR_OPERATOR_H
#define _LCNN_IR_OPERATOR_H

#include "ir/op/op.h"
#include "ir/operand.h"
#include "ir/operator_parameter.h"
#include "ir/operator_attribute.h"

namespace lcnn {

// class Op;
struct Operator
{
    std::string name; // 计算节点名称 如conv1
    std::string type; // 计算节点对应的类型 如nn.Conv2d
    std::shared_ptr<Op> op; // 对应的具体计算op

    bool has_forward = false; // 用于生成计算图中的算子拓扑序，代表该算子是否forward（计算过）

    // 输出Operator节点名字对应关系
    std::map<std::string, std::shared_ptr<Operator>> output_operators;

    // 操作数相关
    // 输出节点名称
    std::vector<std::string> output_names;
    // 输出操作数
    std::shared_ptr<Operand> output_operand;
    // 输入操作数，string代表producer op->name
    std::map<std::string, std::shared_ptr<Operand>> input_operands; // 输入操作数
    // 输入操作数序列
    std::vector<std::shared_ptr<Operand>> input_operands_seq; // 输入操作数序列

    // 算子参数——Param 如stride、padding
    std::map<std::string, OperatorParameter*> params;

    // 算子属性——Attribute 如weight相关
    std::map<std::string, std::shared_ptr<OperatorAttribute>> attribute;

    // 析构
    ~Operator() {
        for (auto& param : this->params) {
            if (param.second != nullptr) {
                delete param.second;
                param.second = nullptr;
            }
        }
    }
};


} // namespace lcnn

#endif // _LCNN_IR_OPERATOR_H
