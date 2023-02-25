/**
 * op_register.cc
 * 用于注册算子使用
 * [by lgx 2023-02-22]
*/

#include "ir/op/op_register.h"
#include "ir/operator.h"

namespace lcnn {

// 由op_type 添加该op的Creator至注册表
// 如名字所示 注册Creator
void OpRegisterer::RegisterCreator(const std::string &op_type,
                                   const OpCreator &creator) {
    if (creator == nullptr) {
        std::cout << "[Error] Op Creator is nullptr" << std::endl;
        return;
    }
    // 获得注册表
    OpRegistry &registry = Registry();
    if (registry.count(op_type) != 0) {
        std::cout << "[Error] Op type: " << op_type << " has already registered!" << std::endl;
        return;
    }
    // 注册op
    registry.insert({op_type, creator});
}

OpRegisterer::OpRegistry &OpRegisterer::Registry() {
    static OpRegistry *kRegistry = new OpRegistry();
    if (kRegistry == nullptr) {
        std::cout << "[Error] Global register init failed!" << std::endl;
    }
    return *kRegistry;
}

// 由Operator创建对应的op，步骤是从registry中获取OpCreator方法，然后调用OpCreator(GetInstance)方法生成Op
std::shared_ptr<Op> OpRegisterer::CreateOp(
    const std::shared_ptr<Operator> &con_operator) {
    // 获得注册表
    OpRegistry &registry = Registry();
    const std::string &op_type = con_operator->type;
    if (registry.count(op_type) <= 0) {
        std::cout << "[Error] Can not find the op type: " << op_type << std::endl;
        return nullptr;
    }
    // 获得OpCreator方法
    const auto &creator = registry.find(op_type)->second;
    if (!creator) {
        std::cout << "[Error] Op creator is empty : " << op_type << std::endl;
        return nullptr;
    }
    std::shared_ptr<Op> op;
    const auto &status = creator(con_operator, op);
    if (status != ParseParameterAttrStatus::kParameterAttrParseSuccess) {
        std::cout << "[Error] Create the op: " << op_type << " failed, error code: " << int(status) << std::endl;
        return nullptr;
    }
    return op;
}

} // namespace lcnn
