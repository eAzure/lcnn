/**
 * op.h
 * 主要目的用于封装具体的算数逻辑
 * [by lgx 2023-02-17]
*/

#ifndef _LCNN_IR_OP_H
#define _LCNN_IR_OP_H

#include <string>
#include "tensor/tensor.h"
#include "utils/status_code.h"
// TODO(搞清楚这里的循环引用关系)
/**
 * 目前大概是这么一个关系：对于A、B互相引用的类来说，可以在一方使用前向声明（forward declaration），另一方采用#include 头文件的方式
 * 假设B采用#include头文件的方式包含A，这说明A的编译顺序应该在B之前，这样的话B是可以看到A的，但是对于A来说此时看不到B，所以此时可以先前向声明一个B，相当于占位符
 * 然后这个时候A只能使用指向B类型的指针，不能进行实例化，因为此时并不知道B的具体成员等
 * 总结：
 *  A不需要#include "B.h" 因为此时A看不到B，只需要在定义class A之前前向声明B(class B;)
 *  B中只需要#include "A.h" 此时不需要对B进行声明
 * A.cc 和 B.cc 中只需要#include "B.h"即可
 * 在本项目中，Op类相当于A，Operator相当于B
 * */
// #include "ir/operator.h"

namespace lcnn {

class Operator;
class Op {
public:
    explicit Op(std::string name):_name(std::move(name)) {}
    virtual ~Op() = default;
    // 前向计算，具体执行计算的函数
    virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs);
    // 前向计算，用于计算图遍历时调用，里面会调用具体的计算接口
    virtual InferStatus Forward();
    // 返回Op _name
    virtual const std::string &op_name() const {
        return this->_name;
    }

    // 设置Operator
    void set_operator(const std::shared_ptr<Operator>& con_operator);
private:
    std::string _name; // Op名称
    std::weak_ptr<Operator> _operator; // 绑定的计算图中Operator节点 // 这里就会导致相互引用的问题
};

} // namespace lcnn

#endif // _LCNN_IR_OP_H