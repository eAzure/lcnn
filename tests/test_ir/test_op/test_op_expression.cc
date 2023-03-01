/**
 * test_op_expression.cc
 * 测试expression op
 * by lgx [2023-02-26]
*/

#include "gtest/gtest.h"
#include "ir/op/op_register.h"
#include "ir/op/ops/expression.h"

// 测试expression op的注册
TEST(test_op, expression_op_register) {
    const auto &registry = lcnn::OpRegisterer::Registry();
    
    std::shared_ptr<lcnn::Operator> runtime_operator = std::make_shared<lcnn::Operator>();
    runtime_operator->name = "expression";
    runtime_operator->type = "pnnx.Expression";
    // [TODO] 这样不对哈，会导致Segmentation fault 因为Operator中存在析构函数，会释放param指针，所以这里必须new出来
    // lcnn::OperatorParameterString* param;
    lcnn::OperatorParameterString* param = new lcnn::OperatorParameterString;
    param->value = "@1";
    runtime_operator->params["expr"] = param;

    std::shared_ptr<lcnn::Op> expression_op = lcnn::OpRegisterer::CreateOp(runtime_operator);
    runtime_operator->op = expression_op;
    ASSERT_EQ(expression_op->op_name(), "Expression");
}

// 测试expression解析
TEST(test_op, expression_op_parser) {
    std::shared_ptr<lcnn::Operator> runtime_operator = std::make_shared<lcnn::Operator>();
    runtime_operator->name = "expression";
    runtime_operator->type = "pnnx.Expression";

    lcnn::OperatorParameterString* param = new lcnn::OperatorParameterString;
    param->value = "add(@0, mul(@1, add(@2, mul(@3, @4))))";
    runtime_operator->params["expr"] = param;

    std::shared_ptr<lcnn::Op> expression_op = lcnn::OpRegisterer::CreateOp(runtime_operator);
    runtime_operator->op = expression_op;

    auto true_expression_op = std::dynamic_pointer_cast<lcnn::ExpressionOp>(expression_op);

    // 输入数据1 batch_size 1
    std::shared_ptr<lcnn::Tensor<float>> input1 = std::make_shared<lcnn::Tensor<float>>(1, 3, 1);
    input1->Ones();
    // 输入数据2 batch_size 2
    std::shared_ptr<lcnn::Tensor<float>> input2 = std::make_shared<lcnn::Tensor<float>>(1, 3, 1);
    input2->Fill(2.f);
    // 输入数据3 batch_size 3
    std::shared_ptr<lcnn::Tensor<float>> input3 = std::make_shared<lcnn::Tensor<float>>(1, 3, 1);
    input3->Fill(3.f);
    // 输入数据4 batch_size 4
    std::shared_ptr<lcnn::Tensor<float>> input4 = std::make_shared<lcnn::Tensor<float>>(1, 3, 1);
    input4->Fill(3.f);
    // 输入数据5 batch_size 5
    std::shared_ptr<lcnn::Tensor<float>> input5 = std::make_shared<lcnn::Tensor<float>>(1, 3, 1);
    input5->Fill(3.f);

    // 1 + (2 * (3 + 3*3)) = 25
    std::vector<std::shared_ptr<lcnn::Tensor<float>>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    inputs.push_back(input4);
    inputs.push_back(input5);
    std::vector<std::shared_ptr<lcnn::Tensor<float>>> outputs(1);

    lcnn::InferStatus infer_status =  expression_op->Forward(inputs, outputs);

    ASSERT_EQ(infer_status, lcnn::InferStatus::kInferSuccess);
    
    for (int i=0;i<outputs.size();i++) {
        std::shared_ptr<lcnn::Tensor<float>> output_data = outputs.at(i);
        output_data->Show();
    }
}