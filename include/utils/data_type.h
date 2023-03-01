/**
 * data_type.h
 * 相关数据类型
 * [by lgx 2023-02-25]
*/

#ifndef _LCNN_UTILS_DATA_TYPE_H
#define _LCNN_UTILS_DATA_TYPE_H

namespace lcnn {

// 操作数对应的数据类型
enum class OperandDataType {
    kTypeUnknown = 0, // 类型未知
    kTypeFloat32 = 1, // float32
};

// 操作数参数对应的数据类型
enum class OperatorParameterType {
    kTypeUnknown = 0, // 类型未知
    kTypeBool = 1, // bool
    kTypeInt = 2, // int
    kTypeFloat = 3, // float
    kTypeString = 4, // string
};

} // namespace lcnn

#endif // _LCNN_UTILS_DATA_TYPE_H