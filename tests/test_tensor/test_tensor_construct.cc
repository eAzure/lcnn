/**
 * test_tensor_construct.cc
 * [by lgx 2023-02-11]
*/

#include "gtest/gtest.h"
#include "tensor/tensor.h"

TEST(test_tensor, tensor_construct1) {
    lcnn::Tensor<float> f1(3, 224, 224);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f1.size(), 3 * 224 * 224);
}

TEST(test_tensor, tensor_construct2) {
    std::vector<uint32_t> shapes{3, 224, 224};
    lcnn::Tensor<float> f1(shapes);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f1.size(), 3 * 224 * 224);
}

TEST(test_tensor, tensor_copy_construct) {
    lcnn::Tensor<float> f1(3, 224, 224);
    f1.Rand();
    lcnn::Tensor<float> f2(f1);
    ASSERT_EQ(f2.channels(), 3);
    ASSERT_EQ(f2.rows(), 224);
    ASSERT_EQ(f2.cols(), 224);
    ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
    
    lcnn::Tensor<float> f3 = f1;
    ASSERT_EQ(f3.channels(), 3);
    ASSERT_EQ(f3.rows(), 224);
    ASSERT_EQ(f3.cols(), 224);
    ASSERT_TRUE(arma::approx_equal(f3.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, tensor_copy_assignment) {
    lcnn::Tensor<float> f1(3, 224, 224);
    f1.Rand();
    lcnn::Tensor<float> f2(3, 2, 1);
    f2 = f1;
    ASSERT_EQ(f2.channels(), 3);
    ASSERT_EQ(f2.rows(), 224);
    ASSERT_EQ(f2.cols(), 224);
    ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, tensor_move_construct) {
    lcnn::Tensor<float> f1(3, 224, 224);
    lcnn::Tensor<float> f2(std::move(f1));
    ASSERT_EQ(f2.channels(), 3);
    ASSERT_EQ(f2.rows(), 224);
    ASSERT_EQ(f2.cols(), 224);
    // 这个时候f1对应的data实际执行应为nullptr
    ASSERT_EQ(f1.raw_ptr(), nullptr);
}

TEST(test_tensor, tensor_move_assignment) {
    lcnn::Tensor<float> f1(3, 224, 224);
    lcnn::Tensor<float> f2(3, 2, 1);
    f2 = std::move(f1);
    ASSERT_EQ(f2.channels(), 3);
    ASSERT_EQ(f2.rows(), 224);
    ASSERT_EQ(f2.cols(), 224);
    // 这个时候f1对应的data实际执行应为nullptr
    ASSERT_EQ(f1.raw_ptr(), nullptr);
}

TEST(test_tensor, tensor_set_data) {
    lcnn::Tensor<float> f1(3, 224, 224);
    arma::fcube cube1(224, 224, 3);
    cube1.randn();
    f1.set_data(cube1);

    ASSERT_TRUE(arma::approx_equal(f1.data(), cube1, "absdiff", 1e-4));
}