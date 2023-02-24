/**
 * status_code.h
 * 各种状态码
 * [by lgx 2023-02-17]
*/

#ifndef _LCNN_UTILS_STATUS_CODE_H
#define _LCNN_UTILS_STATUS_CODE_H

namespace lcnn {

// 用于推理状态
enum class InferStatus {
    kInferUnknown = -1, // 未知状态，一般用于op未实现
    kInferSuccess = 0, // 推理成功状态
    
    kInferFailedInputEmpty = 1, // 输入feature为空
    kInferFailedInputOutputSizeNotEqual = 2, // 输入输出size不相等

};

// 用于解析params 和 attrs
enum class ParseParameterAttrStatus {
    kParameterMissingUnknown = -1,
    kParameterAttrParseSuccess = 0, // 解析成功
};

} // namespace lcnn

#endif // _LCNN_UTILS_STATUS_CODE_H
