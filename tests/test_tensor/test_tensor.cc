/**
 * test_tensor.cc
 * [by lgx 2023-02-12]
*/
#include "gtest/gtest.h"
#include "tensor/tensor.h"

// 测试empty()
TEST(test_tensor, tensor_empty) {
    lcnn::Tensor<float> t1;
    ASSERT_EQ(t1.empty(), true);

    lcnn::Tensor<float> t2(1, 2, 3);
    ASSERT_EQ(t2.empty(), false);
}

/********* 测试一些访存操作 *********/
// 测试index(offset)
TEST(test_tensor, tensor_index) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    tensor.index(6) = 7.2f;
    ASSERT_EQ(tensor.index(6), 7.2f);
}
// 测试at(channel)
TEST(test_tensor, tensor_at_channel) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    arma::fmat mat(4, 5);
    mat.fill(1.2f);
    tensor.at(0) = mat;
    ASSERT_TRUE(arma::approx_equal(mat, tensor.at(0), "absdiff", 1e-4));
}
// 测试at(channel, row, col)
TEST(test_tensor, tensor_at_channel_row_col) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    tensor.at(0, 1, 2) = 2.2f;
    ASSERT_EQ(tensor.at(0, 1, 2), 2.2f);
}

/********* 测试数据填充 *********/
// 测试Fill(float)
TEST(test_tensor, tensor_fill_float) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    tensor.Fill(1.3f);
    for (int c=0;c<tensor.channels();c++) {
        for (int h=0;h<tensor.rows();h++) {
            for (int w=0;w<tensor.cols();w++) {
                ASSERT_EQ(tensor.at(c, h, w), 1.3f);
            }
        }
    }
}
// 测试Fill(vector<float>)
TEST(test_tensor, tensor_fill_vector) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    std::vector<float> values;
    for (int i=0;i<tensor.size();i++) {
        values.push_back(float(i));
    }
    tensor.Fill(values);
    for (int c=0;c<tensor.channels();c++) {
        for (int h=0;h<tensor.rows();h++) {
            for (int w=0;w<tensor.cols();w++) {
                ASSERT_EQ(tensor.at(c, h, w), 
                          values[c * tensor.rows() * tensor.cols() 
                                 + h * tensor.cols() + w]);
            }
        }
    }
}
// 测试Ones()
TEST(test_tensor, tensor_ones) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    tensor.Ones();
    for (int i=0;i<tensor.size();i++) {
        ASSERT_EQ(tensor.index(i), 1.f);
    }
}

// 测试Show()
TEST(test_tensor, tensor_print) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    std::vector<float> values;
    for (int i=0;i<tensor.size();i++) {
        values.push_back(float(i));
    }
    tensor.Fill(values);
    tensor.Show();
}

/********* 测试张量变换 *********/
// 测试Padding
TEST(test_tensor, tensor_padding) {
    lcnn::Tensor<float> tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.Fill(1.f);
    tensor.Padding({2, 2, 2, 2}, 2.f);
    ASSERT_EQ(tensor.rows(), 8);
    ASSERT_EQ(tensor.cols(), 9);

    for (int c=0;c<tensor.channels();c++) {
        for (int h=0;h<tensor.rows();h++) {
            for (int w=0;w<tensor.cols();w++) {
                if (h<=1 || w<=1) {
                    ASSERT_EQ(tensor.at(c, h, w), 2.f);
                } else if (h>=tensor.rows()-2 || w>=tensor.cols()-2) {
                    ASSERT_EQ(tensor.at(c, h, w), 2.f);
                } else {
                    ASSERT_EQ(tensor.at(c, h, w), 1.f);
                }
            }
        }
    }
}
// 测试Flatten
TEST(test_tensor, tensor_flatten) {
    lcnn::Tensor<float> tensor(3, 3, 3);
    std::vector<float> values;
    for (int i=0;i<tensor.size();i++) {
        values.push_back(float(i));
    }
    tensor.Fill(values);
    tensor.Show();
    tensor.Flatten();
    ASSERT_EQ(tensor.channels(), 1);
    ASSERT_EQ(tensor.rows(), tensor.size());
    ASSERT_EQ(tensor.cols(), 1);
    tensor.Show();
} 
// 测试Transform
TEST(test_tensor, tensor_transform) {
    lcnn::Tensor<float> tensor(3, 3, 3);
    ASSERT_EQ(tensor.empty(), false);
    tensor.Fill(1.f);
    tensor.Transform([](const float& value)->float {
        return 1.f;
    });
    for (int i=0;i<tensor.size();i++) {
        ASSERT_EQ(tensor.index(i), 1.f);
    }
}

/********* 测试shape变换 *********/
// 测试ReRawShape
TEST(test_tensor, tensor_rerawshape_1) {
    lcnn::Tensor<float> tensor(2, 3, 4);
    // for (int i=0;i<tensor.size();i++) {
    //     tensor.index(i) = i;
    // }

    int index = 0;
    for (int c=0;c<tensor.channels();c++) {
        for (int h=0;h<tensor.rows();h++) {
            for (int w=0;w<tensor.cols();w++) {
                tensor.at(c, h, w) = index++;
            }
        }
    }

    tensor.ReRawShape({24});
    const auto& shapes1 = tensor.raw_shapes();
    ASSERT_EQ(shapes1.size(), 1);
    ASSERT_EQ(shapes1.at(0), 24);

    tensor.ReRawShape({4, 6});
    const auto& shapes2 = tensor.raw_shapes();
    ASSERT_EQ(shapes2.size(), 2);
    ASSERT_EQ(shapes2.at(0), 4);
    ASSERT_EQ(shapes2.at(1), 6);

    tensor.ReRawShape({4, 3, 2});
    const auto& shapes3 = tensor.raw_shapes();
    ASSERT_EQ(shapes3.size(), 3);
    ASSERT_EQ(shapes3.at(0), 4);
    ASSERT_EQ(shapes3.at(1), 3);
    ASSERT_EQ(shapes3.at(2), 2);
}
// 列主序
TEST(test_tensor, tensor_rerawshape_2) {
    arma::fmat t1 =
        "0, 2;"
        "1, 3";
    arma::fmat t2 = 
        "4, 5;"
        "6, 7";

    lcnn::Tensor<float> data(2, 2, 2);
    data.at(0) = t1;
    data.at(1) = t2;
    data.Show();
    data.ReRawShape({8});
    data.Show();
}
// 测试ReRawView
TEST(test_tensor, tensor_rerawview_1) {
    lcnn::Tensor<float> tensor(2, 3, 4);
    // for (int i=0;i<tensor.size();i++) {
    //     tensor.index(i) = i;
    // }

    int index = 0;
    for (int c=0;c<tensor.channels();c++) {
        for (int h=0;h<tensor.rows();h++) {
            for (int w=0;w<tensor.cols();w++) {
                tensor.at(c, h, w) = index++;
            }
        }
    }

    tensor.ReRawView({24});
    const auto& shapes1 = tensor.raw_shapes();
    ASSERT_EQ(shapes1.size(), 1);
    ASSERT_EQ(shapes1.at(0), 24);

    tensor.ReRawView({4, 6});
    const auto& shapes2 = tensor.raw_shapes();
    ASSERT_EQ(shapes2.size(), 2);
    ASSERT_EQ(shapes2.at(0), 4);
    ASSERT_EQ(shapes2.at(1), 6);

    tensor.ReRawView({4, 3, 2});
    const auto& shapes3 = tensor.raw_shapes();
    ASSERT_EQ(shapes3.size(), 3);
    ASSERT_EQ(shapes3.at(0), 4);
    ASSERT_EQ(shapes3.at(1), 3);
    ASSERT_EQ(shapes3.at(2), 2);
}
// 列主序
TEST(test_tensor, tensor_rerawview_2) {
    arma::fmat t1 =
        "0, 2;"
        "1, 3";
    arma::fmat t2 = 
        "4, 5;"
        "6, 7";

    lcnn::Tensor<float> data(2, 2, 2);
    data.at(0) = t1;
    data.at(1) = t2;
    data.Show();
    data.ReRawView({8});
    data.Show();
}

// 测试Clone
TEST(test_tensor, tensor_clone) {
    lcnn::Tensor<float> tensor(3, 3, 3);
    ASSERT_EQ(tensor.empty(), false);
    tensor.Rand();

    const auto& clone_tensor = tensor.Clone();
    ASSERT_NE(clone_tensor->raw_ptr(), tensor.raw_ptr());
    ASSERT_EQ(clone_tensor->size(), tensor.size());
    for (int i=0;i<tensor.size();i++) {
        ASSERT_EQ(clone_tensor->index(i), tensor.index(i));
    }
}

/********* 处理Tensor的相关库 *********/
// 测试TensorCreate
TEST(test_tensor, tensor_create) {
    const std::shared_ptr<lcnn::Tensor<float>>& tensor_ptr1 = lcnn::TensorCreate(3, 224, 224);
    ASSERT_EQ(tensor_ptr1->empty(), false);
    ASSERT_EQ(tensor_ptr1->channels(), 3);
    ASSERT_EQ(tensor_ptr1->rows(), 224);
    ASSERT_EQ(tensor_ptr1->cols(), 224);

    const std::shared_ptr<lcnn::Tensor<float>>& tensor_ptr2 = lcnn::TensorCreate({3, 224, 224});
    ASSERT_EQ(tensor_ptr2->empty(), false);
    ASSERT_EQ(tensor_ptr2->channels(), 3);
    ASSERT_EQ(tensor_ptr2->rows(), 224);
    ASSERT_EQ(tensor_ptr2->cols(), 224);
}
// 测试TensorIsSame
TEST(test_tensor, tensor_is_same) {
    std::shared_ptr<lcnn::Tensor<float>> tensor1 = std::make_shared<lcnn::Tensor<float>>(3, 32, 32);
    tensor1->Fill(2.f);
    std::shared_ptr<lcnn::Tensor<float>> tensor2 = std::make_shared<lcnn::Tensor<float>>(3, 32, 32);
    tensor2->Fill(2.f);
    std::shared_ptr<lcnn::Tensor<float>> tensor3 = std::make_shared<lcnn::Tensor<float>>(3, 32, 32);
    tensor3->Fill(3.f);

    ASSERT_EQ(lcnn::TensorIsSame(tensor1, tensor2), true);
    ASSERT_EQ(lcnn::TensorIsSame(tensor1, tensor3), false);
}
// 测试TensorBroadCast
TEST(test_tensor, tensor_broadcast) {
    const std::shared_ptr<lcnn::Tensor<float>>& tensor1 =
        lcnn::TensorCreate(3, 224, 224);
    const std::shared_ptr<lcnn::Tensor<float>>& tensor2 =
        lcnn::TensorCreate(3, 1, 1);
    tensor2->Rand();
    const auto& new_tensor_tuple = lcnn::TensorBroadCast(tensor1, tensor2);
    const auto& new_tensor1 = std::get<0>(new_tensor_tuple);
    const auto& new_tensor2 = std::get<1>(new_tensor_tuple);

    ASSERT_EQ(new_tensor2->channels(), 3);
    ASSERT_EQ(new_tensor2->rows(), 224);
    ASSERT_EQ(new_tensor2->cols(), 224);

    for (uint32_t c=0;c<tensor2->channels();c++) {
        float temp = tensor2->index(c);
        const auto& channel_data = new_tensor2->at(c);
        for (uint32_t i=0;i<channel_data.size();i++) {
            ASSERT_EQ(channel_data.at(i), temp);
        }
    }
}
// 测试TensorElementAdd
TEST(test_tensor, tensor_element_add_1) {
    const auto& tensor1 = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    tensor1->Fill(1.f);
    const auto& tensor2 = std::make_shared<lcnn::Tensor<float>>(3, 1, 1);
    tensor2->Fill(2.f);
    const auto& tensor3 = lcnn::TensorElementAdd(tensor1, tensor2);
    for (int i=0;i<tensor3->size();i++) {
        ASSERT_EQ(tensor3->index(i), 3.f);
    }
}
TEST(test_tensor, tensor_element_add_2) {
    const auto& tensor1 = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    tensor1->Fill(2.f);
    const auto& tensor2 = std::make_shared<lcnn::Tensor<float>>(3, 1, 1);
    tensor2->Fill(3.f);
    const auto& tensor3 = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    lcnn::TensorElementAdd(tensor1, tensor2, tensor3);
    for (int i=0;i<tensor3->size();i++) {
        ASSERT_EQ(tensor3->index(i), 5.f);
    }
}
// 测试TensorElementMultiply
TEST(test_tensor, tensor_element_multiply_1) {
    const auto& tensor1 = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    tensor1->Fill(2.f);
    const auto& tensor2 = std::make_shared<lcnn::Tensor<float>>(3, 1, 1);
    tensor2->Fill(3.f);
    const auto& tensor3 = lcnn::TensorElementMultiply(tensor1, tensor2);
    for (int i=0;i<tensor3->size();i++) {
        ASSERT_EQ(tensor3->index(i), 6.f);
    }
}
TEST(test_tensor, tensor_element_multiply_2) {
    const auto& tensor1 = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    tensor1->Fill(2.f);
    const auto& tensor2 = std::make_shared<lcnn::Tensor<float>>(3, 1, 1);
    tensor2->Fill(3.f);
    const auto& tensor3 = std::make_shared<lcnn::Tensor<float>>(3, 224, 224);
    lcnn::TensorElementMultiply(tensor1, tensor2, tensor3);
    for (int i=0;i<tensor3->size();i++) {
        ASSERT_EQ(tensor3->index(i), 6.f);
    }
}