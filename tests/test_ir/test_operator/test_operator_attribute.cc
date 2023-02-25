/**
 * test_operator_attribute.cc
 * 测试OperatorAttribute
 * by lgx [2023-02-25]
*/

#include "gtest/gtest.h"
#include "ir/operator_attribute.h"

// 测试获取weight_data
TEST(test_operator, attribute_weight_data_get) {
    lcnn::OperatorAttribute operator_attr;
    operator_attr.type = lcnn::OperandDataType::kTypeFloat32;
    std::vector<char> weight_data;
    for (int i=0;i<32;i++) {
        weight_data.push_back(0);
    }
    operator_attr.weight_data = weight_data;

    const auto &get_weight_data = operator_attr.get<float>();
    ASSERT_EQ(get_weight_data.size(), 8);
    for (int i=0;i<8;i++) {
        std::cout << get_weight_data[i] << " ";
    }
    std::cout << std::endl;
}

// 测试shape属性信息
TEST(test_operator, attribute_shape) {
    lcnn::OperatorAttribute operator_attr;
    operator_attr.type = lcnn::OperandDataType::kTypeFloat32;
    operator_attr.shape = std::vector<uint32_t>{3, 32, 32};
    ASSERT_EQ(operator_attr.shape.at(0), 3);
    ASSERT_EQ(operator_attr.shape.at(1), 32);
    ASSERT_EQ(operator_attr.shape.at(2), 32);
}