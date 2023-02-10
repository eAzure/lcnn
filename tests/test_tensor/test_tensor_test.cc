/**
 * first test for tensor
 * [by lgx 2023-02-10]
*/

#include "gtest/gtest.h"
#include "tensor/tensor.h"

TEST(test_tensor, first) {
    lcnn::Tensor<float> tensor(3, 224, 224);
    tensor.test();
}