/**
 * operator_parameter.h
 * 算子节点参数，如stride padding，还有expression op中的expr参数等
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

// 对应bool类型
struct OperatorParameterBool : public OperatorParameter
{
    OperatorParameterBool() : OperatorParameter(OperatorParameterType::kTypeBool) {}
    bool value = false;
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

// 对应string类型
struct OperatorParameterString : public OperatorParameter
{
    OperatorParameterString() : OperatorParameter(OperatorParameterType::kTypeString) {}
    std::string value = "";
};

// 对应array<int>类型
struct OperatorParameterIntArray : public OperatorParameter
{
    OperatorParameterIntArray() : OperatorParameter(OperatorParameterType::kTypeIntArray) {}
    std::vector<int> value;
};

// 对应array<float>类型
struct OperatorParameterFloatArray : public OperatorParameter
{
    OperatorParameterFloatArray() : OperatorParameter(OperatorParameterType::kTypeFloatArray) {}
    std::vector<float> value;
};

// 对应array<string>类型
struct OperatorParameterStringArray : public OperatorParameter
{
    OperatorParameterStringArray() : OperatorParameter(OperatorParameterType::kTypeStringArray) {}
    std::vector<std::string> value; 
};




} // namespace lcnn

#endif // _LCNN_IR_OPERATOR_PARAMETER_H