/**
 * op_register.h
 * 用于注册op使用：本质上是根据operator的op_type生成对应的op
 * [by lgx 2023-02-22]
*/

#ifndef _LCNN_IR_OP_REGISTER_H
#define _LCNN_IR_OP_REGISTER_H

#include "op.h"
#include "ir/operator.h"

namespace lcnn {

class OpRegisterer {
public:
    // 算子的构造器
    typedef ParseParameterAttrStatus (*OpCreator)(const std::shared_ptr<Operator> &con_operator,
                                                std::shared_ptr<Op> &op);
    // 定义注册表数据结构
    typedef std::map<std::string, OpCreator> OpRegistry;
    // 由op_type 添加该op的Creator至注册表中，用于op声明时
    static void RegisterCreator(const std::string &op_type, const OpCreator &creator);
    // 由operator初始化对应的op，用于遍历计算图生成计算节点时
    static std::shared_ptr<Op> CreateOp(const std::shared_ptr<Operator> &con_operator);
    // 生成注册表
    static OpRegistry &Registry();
};

// 封装注册器
class OpRegistererWrapper {
public:
    OpRegistererWrapper(const std::string &op_type, const OpRegisterer::OpCreator &creator) {
        OpRegisterer::RegisterCreator(op_type, creator);
    }
};

} // namespace lcnn

#endif // _LCNN_IR_OP_REGISTER_H
