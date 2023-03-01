/**
 * expression.cc
 * [by lgx 2023-02-26]
*/

#include "ir/op/ops/expression.h"
#include "ir/op/op_register.h"
#include <stack>

namespace lcnn {

// 这里的数据组织方式是输入操作数的数量应该与@data的数量个数一致，然后每一个的batch_size大小应该一致
// 并且该大小应该与output的batch_size大小一致，且连续存储在inputs里面，比如说batch_size大小为5
// 那么inputs前5个代表的是@0，再接着5个代表的是@1
InferStatus ExpressionOp::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    if (inputs.empty()) {
        std::cout << "[Error] The input feature of expression op is empty!" << std::endl;
        return InferStatus::kInferFailedInputEmpty;
    }
    if (outputs.empty()) {
        std::cout << "[Error] The output of expression op is empty!" << std::endl;
        return InferStatus::kInferFailedInputOutputSizeNotEqual;
    }

    if (this->_parser == nullptr) {
        std::cout << "[Error] The parser of expression is nullptr!" << std::endl;
        return InferStatus::kInferFailedExpressionParserIsNull;
    }
    // 词法分析
    this->_parser->Tokenizer(false);
    
    // 可以查看词法分析的结果
    // std::cout << "Tokens: " << std::endl;
    // for (auto token:this->_parser->token_strs()) {
    //     std::cout << token << " ";
    // }
    // std::cout << std::endl;

    const auto &expressions = this->_parser->tokens();
    if (expressions.empty()) {
        std::cout << "[Error] Tokenize failed!" << std::endl;
        return InferStatus::kInferFailed;
    }

    for (uint32_t i=0;i<inputs.size();i++) {
        if (inputs.at(i) == nullptr || inputs.at(i)->empty()) {
            std::cout << "[Error] The input feature of expression op is empty!" << std::endl;
            return InferStatus::kInferFailedInputEmpty;
        }
    }

    const uint32_t batch_size = outputs.size(); // 注意这里是outputs.size() 而不是inputs的
    // 获得逆波兰式的Token序列，准备操作数等
    std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack; // 一个元素对应一个操作数@data
    const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->_parser->Generate();
    for (const auto &token_node : token_nodes) {
        if (token_node->num_index >= 0) {
            // 代表Data 如@0 @1 @2 这里的次序对应着inputs里面的输入数据次序
            // 每一个都具有batch_size个Tensor
            // std::cout << "data: " << token_node->num_index << std::endl;
            uint32_t start_pos = token_node->num_index * batch_size;
            std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
            for (uint32_t i=0;i<batch_size;i++) {
                input_token_nodes.push_back(inputs.at(start_pos+i));
            }
            op_stack.push(input_token_nodes);
        } else {
            // 代表add mul
            // std::cout << "op: " << token_node->num_index << std::endl;
            // std::cout << token_node->left->num_index << ", " << token_node->right->num_index << std::endl;
            const int32_t op = token_node->num_index;
            if (op_stack.size() < 2) {
                std::cout << "[Error] The number of operand in the expression is less than two!" << std::endl;
                return InferStatus::kInferFailed;
            }
            // 获得第一个操作数            
            std::vector<std::shared_ptr<Tensor<float>>> input_node1 = op_stack.top();
            // std::cout << "input_node1.size(): " << input_node1.size() << ", batch_size: " << batch_size << std::endl;
            if (input_node1.size() != batch_size) {
                std::cout << "[Error] The input_node1.size is not equal to batch_size!" << std::endl;
                return InferStatus::kInferFailed;
            }
            op_stack.pop();
            // 获得第二个操作数
            std::vector<std::shared_ptr<Tensor<float>>> input_node2 = op_stack.top();
            // std::cout << "input_node2.size(): " << input_node2.size() << ", batch_size: " << batch_size << std::endl;
            if (input_node2.size() != batch_size) {
                std::cout << "[Error] The input_node2.size is not equal to batch_size!" << std::endl;
                return InferStatus::kInferFailed;
            }
            op_stack.pop();
            std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
            for (uint32_t i=0;i<batch_size;i++) {
                if (op == int(TokenType::kTypeAdd)) {
                    output_token_nodes.at(i) = TensorElementAdd(input_node1.at(i), input_node2.at(i));
                } else if (op == int(TokenType::kTypeMul)) {
                    output_token_nodes.at(i) = TensorElementMultiply(input_node1.at(i), input_node2.at(i));
                } else {
                    std::cout << "[Error] Unknown operator type: " << op << std::endl;
                }
            }
            op_stack.push(output_token_nodes);
        }
    }
    if (op_stack.size() != 1) {
        std::cout << "[Error] Executing expression error! Op_stack.size is not equal to 1." << std::endl;
    }
    std::vector<std::shared_ptr<Tensor<float>>> output_node = op_stack.top();
    op_stack.pop();
    for (int i=0;i<batch_size;i++) {
        outputs.at(i) = output_node.at(i);
    }
    return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ExpressionOp::GetInstance(const std::shared_ptr<Operator> &con_operator,
                                                   std::shared_ptr<Op> &expression_op) {
    if (con_operator == nullptr) {
        std::cout << "[Error] Expression operator is nullptr!" << std::endl;
    }
    const auto &params = con_operator->params;
    if (params.find("expr") == params.end()) {
        std::cout << "[Error] 'expr' parameter is missing!" << std::endl;
        return ParseParameterAttrStatus::kParameterMissingExpr;
    }
    const auto &statement_param = dynamic_cast<OperatorParameterString *>(params.at("expr"));
    if (statement_param == nullptr) {
        std::cout << "[Error] 'expr' parameter is nullptr!" << std::endl;
        return ParseParameterAttrStatus::kParameterMissingExpr;       
    }
    if (statement_param->type != OperatorParameterType::kTypeString) {
        std::cout << "[Error] 'expr' parameter's type is not string!" << std::endl;
        return ParseParameterAttrStatus::kParameterMissingExpr;
    }
    // 从这里也可以看出来，operator对于op的创建可以提供一些信息，这里提供了params里面的expr的信息
    expression_op = std::make_shared<ExpressionOp>(statement_param->value);
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

// 算子注册
OpRegistererWrapper kExpressionGetInstance("pnnx.Expression", ExpressionOp::GetInstance);

} // namespace lcnn