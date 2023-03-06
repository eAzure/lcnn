/**
 * test_op_register.cc
 * 测试op注册与operator对应的op生成
 * by lgx [2023-02-23]
*/

#include "gtest/gtest.h"
#include "ir/op/op_register.h"

// 测试单例模式下的注册表
TEST(test_op, test_op_register_registry) {
    const auto &r1 = lcnn::OpRegisterer::Registry();
    const auto &r2 = lcnn::OpRegisterer::Registry();
    ASSERT_EQ(r1, r2);
}

// 测试operator中对应的op生成
TEST(test_op, test_op_register_generateop) {
    const auto &registry = lcnn::OpRegisterer::Registry();
    
    std::shared_ptr<lcnn::Operator> runtime_operator = std::make_shared<lcnn::Operator>();
    runtime_operator->name = "relu";
    runtime_operator->type = "nn.ReLU"; // 这里必须和注册表中注册的一致

    std::shared_ptr<lcnn::Op> relu_op = lcnn::OpRegisterer::CreateOp(runtime_operator);
    runtime_operator->op = relu_op;
    ASSERT_EQ(relu_op->op_name(), "Relu");

    std::shared_ptr<lcnn::Tensor<float>> input = std::make_shared<lcnn::Tensor<float>>(32, 224, 512);
    input->Rand();
    std::vector<std::shared_ptr<lcnn::Tensor<float>>> inputs;
    inputs.push_back(input);
    std::vector<std::shared_ptr<lcnn::Tensor<float>>> outputs(1);

    const auto status = relu_op->Forward(inputs, outputs);
    ASSERT_EQ(status, lcnn::InferStatus::kInferSuccess);

    for (int i=0;i<inputs.size();i++) {
        std::shared_ptr<lcnn::Tensor<float>> input_data = inputs.at(i);
        input_data->Transform([](const float val) {
            return val > 0. ? val:0.;
        });
        std::shared_ptr<lcnn::Tensor<float>> output_data = outputs.at(i);
        ASSERT_EQ(lcnn::TensorIsSame(input_data, output_data), true);
    }

}