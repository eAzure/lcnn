/**
 * status_code.h
 * 各种状态码
 * [by lgx 2023-02-17]
*/

#ifndef _LCNN_UTILS_STATUS_CODE_H
#define _LCNN_UTILS_STATUS_CODE_H

namespace lcnn {

enum class InferStatus {
    kInferUnknown = -1,
    kInferSuccess = 0
};

} // namespace lcnn

#endif // _LCNN_UTILS_STATUS_CODE_H
