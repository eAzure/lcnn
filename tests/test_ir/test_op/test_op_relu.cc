/**
 * test_op_relu.cc
 * [by lgx 2023-02-17]
*/
#include "gtest/gtest.h"
#include "ir/op/ops/relu.h"

TEST(test_op, test_op_relu) {
  std::shared_ptr<lcnn::Tensor<float>> input = std::make_shared<lcnn::Tensor<float>>(32, 224, 512);
  input->Rand();
  std::vector<std::shared_ptr<lcnn::Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<lcnn::Tensor<float>>> outputs(1);

  lcnn::ReluOp relu_op;
  const auto status = relu_op.Forward(inputs, outputs);
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
