/**
 * test_operator_parameter.cc
 * 测试OperatorParameter
 * by lgx [2023-02-25]
*/

#include "gtest/gtest.h"
#include "ir/operator_parameter.h"

TEST(test_operator, parameter) {
    lcnn::OperatorParameter* param = new lcnn::OperatorParameterInt;
    ASSERT_EQ(param->type, lcnn::OperatorParameterType::kTypeInt);
    ASSERT_EQ(dynamic_cast<lcnn::OperatorParameterFloat*>(param), nullptr);
    ASSERT_NE(dynamic_cast<lcnn::OperatorParameterInt*>(param), nullptr);
}