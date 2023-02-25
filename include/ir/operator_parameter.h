/**
 * operator_parameter.h
 * 算子节点参数，如stride padding
 * [by lgx 2023-02-25]
*/

#ifndef _LCNN_IR_OPERATOR_PARAMETER_H
#define _LCNN_IR_OPERATOR_PARAMETER_H

#include "utils/data_type.h"

namespace lcnn {

struct OperatorParameter
{   
    explicit OperatorParameter(OperatorParameterType type = OperatorParameterType::kTypeUnknown) : type(type) {}
    virtual ~OperatorParameter() = default;
    // 对应的参数类型
    OperatorParameterType type = OperatorParameterType::kTypeUnknown;
};

// 对应int类型
struct OperatorParameterInt : public OperatorParameter
{
    OperatorParameterInt() : OperatorParameter(OperatorParameterType::kTypeInt) {}
    int value = 0;
};

// 对应float类型
struct OperatorParameterFloat : public OperatorParameter
{
    OperatorParameterFloat() : OperatorParameter(OperatorParameterType::kTypeFloat) {}
    float value = 0.f;
};




} // namespace lcnn

#endif // _LCNN_IR_OPERATOR_PARAMETER_H