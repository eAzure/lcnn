/**
 * operator_attribute.h
 * 算子节点属性，如weight
 * [by lgx 2023-02-25]
*/

#ifndef _LCNN_IR_OPERATOR_ATTRIBUTE_H
#define _LCNN_IR_OPERATOR_ATTRIBUTE_H

#include "utils/data_type.h"

namespace lcnn {

struct OperatorAttribute
{
    // 权重信息 char类型，之后通过get转换为对应的计算类型
    std::vector<char> weight_data;
    // Operator节点shape信息
    std::vector<uint32_t> shape;
    // 节点数据类型信息
    OperandDataType type = OperandDataType::kTypeUnknown;

    // 从节点中加载权重
    template<class T>
    std::vector<T> get();
};

// 从节点中加载权重
template<class T>
std::vector<T> OperatorAttribute::get() {
    if (weight_data.empty()) {
        std::cout << "[Error] weight_data is empty!" << std::endl;
    }
    if (type == OperandDataType::kTypeUnknown) {
        std::cout << "[Error] type is unknown!" << std::endl;
    }
    std::vector<T> weights;
    switch (type) {
        case OperandDataType::kTypeFloat32: {
            const bool is_float = std::is_same<T, float>::value;
            if (!is_float) {
                std::cout << "[Error] type is not corresponding float!" << std::endl;
                break;
            }
            const uint32_t float_size = sizeof(float);
            if (weight_data.size() % float_size != 0) {
                std::cout << "[Error] weight_data size is not suit for float!" << std::endl;
                break;
            }
            for (uint32_t i=0;i<weight_data.size()/float_size;i++) {
                float weight = *((float*)weight_data.data()+i);
                weights.push_back(weight);
            }
            break;
        }
        default: {
            std::cout << "[Error] Unknown weight data type!" << std::endl;
        }
    }
    return weights;
}

} // namespace lcnn

#endif // _LCNN_IR_OPERATOR_ATTRIBUTE_H