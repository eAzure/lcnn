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

}
