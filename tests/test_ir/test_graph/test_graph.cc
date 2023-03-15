/**
 * test_graph.cc
 * 测试计算图的构建和执行
 * by lgx [2023-03-04]
*/

#include "gtest/gtest.h"
#include "ir/graph.h"

TEST(test_graph, graph_build) {
    lcnn::Graph graph("/home/lcnn/lcnn/tests/resources/relu_add/model_pnnx_mix_batch_10.pnnx.param", "/home/lcnn/lcnn/tests/resources/relu_add/model_pnnx_mix_batch_10.pnnx.bin");
    graph.Build(
        "pnnx_input_0",
        "pnnx_output_0"
    );
    // 准备输入数据
    std::shared_ptr<lcnn::Tensor<float>> input = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    input->Fill(2.f);
    std::vector<std::shared_ptr<lcnn::Tensor<float>>> inputs;
    // inputs.push_back(input);
    int batch_size = 10;
    for (int i=0;i<10;i++) {
        inputs.push_back(input);
    }
    std::vector<std::shared_ptr<lcnn::Tensor<float>>> output = graph.Forward(inputs);
    ASSERT_EQ(output.empty(), false) << "[Error] The output is empty!";
    // output[0]->Show();
    std::cout << "batch_size: " << output.size() << std::endl;
    std::cout << "shape: " << std::endl;
    for (auto shape : output[0]->shapes()) {
        std::cout << shape << " ";
    }
    std::cout << std::endl;
}