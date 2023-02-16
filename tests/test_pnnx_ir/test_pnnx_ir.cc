/**
 * test_pnnx_ir.cc
 * [by lgx 2023-02-16]
*/

#include "gtest/gtest.h"
#include "pnnx_ir/ir.h"

TEST(test_pnnx_ir, test_pnnx) {
    const std::string &param_path = "/home/lcnn/lcnn/tests/resources/add/resnet_add.pnnx.param";
    const std::string &weight_path = "/home/lcnn/lcnn/tests/resources/add/resnet_add.pnnx.bin";

    std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
    int load_result = graph->load(param_path, weight_path);
    ASSERT_EQ(load_result, 0);

    std::vector<pnnx::Operator *> operators = graph->ops;
    ASSERT_FALSE(operators.empty());

    for (const pnnx::Operator *op : operators) {
        if (!op) continue;
        std::cout << "op->name: " << op->name << ", op->type: " << op->type << std::endl;
    }
}