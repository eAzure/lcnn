/**
 * expression_op_parse.cc
 * 用于解析expression_op
 * [by lgx 2023-02-26]
*/

#include "ir/op/expression_op_parse.h"

namespace lcnn {

// 词法分析
// 示例: add(add(add(@0,@1),@1),add(@0,@2))s
void ExpressionOpParser::Tokenizer(bool re_tokenize) {
    // 之前生成过且不重新生成
    if (!re_tokenize && !this->_tokens.empty()) {
        std::cout <<"[Warning] Tokenize has already done!" << std::endl;
        return;
    }
    if (this->_statement.empty()) {
        std::cout << "[Error] The statement is empty!" << std::endl;
        return;
    }
    // 删除空格
    this->_statement.erase(std::remove_if(_statement.begin(), _statement.end(), [](char c) {
        return std::isspace(c);
    }), _statement.end());
    if (this->_statement.empty()) {
        std::cout << "[Error] The statement is empty!" << std::endl;
        return;
    }
    
    for (int i = 0; i< _statement.size();) {
        char c = _statement.at(i);
        if (c=='a') {
            if (i+1 >= _statement.size() || _statement.at(i+1) != 'd') {
                std::cout << "[Error] Parse add token error at: " << i+1 << " in statement " << _statement << std::endl;
                return;
            }
            if (i+2 >= _statement.size() || _statement.at(i+2) != 'd') {
                std::cout << "[Error] Parse add token error at: " << i+2 << " in statement " << _statement << std::endl;
                return;
            }
            Token token(TokenType::kTypeAdd, i, i+3);
            this->_tokens.push_back(token);
            std::string token_operation = std::string(_statement.begin()+i, _statement.begin()+i+3);
            this->_token_strs.push_back(token_operation);
            i = i+3;
        } else if (c=='m') {
            if (i+1 >= _statement.size() || _statement.at(i+1) != 'u') {
                std::cout << "[Error] Parse mul token error at: " << i+1 << " in statement " << _statement << std::endl;
                return;
            }
            if (i+2 >= _statement.size() || _statement.at(i+2) != 'l') {
                std::cout << "[Error] Parse mul token error at: " << i+2 << " in statement " << _statement << std::endl;
                return;
            }
            Token token(TokenType::kTypeMul, i, i+3);
            this->_tokens.push_back(token);
            std::string token_operation = std::string(_statement.begin()+i, _statement.begin()+i+3);
            this->_token_strs.push_back(token_operation);
            i = i+3;
        } else if (c=='@') {
            if (i+1 >= _statement.size() || !std::isdigit(_statement.at(i+1))) {
                std::cout << "[Error] Parse data token error at: " << i+1 << " in statement " << _statement << std::endl;
                return;
            }
            int j = i+1;
            for(; j<_statement.size();j++) {
                if (!std::isdigit(_statement.at(j))) {
                    break;
                }
            }
            Token token(TokenType::kTypeData, i, j);
            this->_tokens.push_back(token);
            std::string token_data = std::string(_statement.begin()+i, _statement.begin()+j);
            this->_token_strs.push_back(token_data);
            i = j;
        } else if (c==',') {
            Token token(TokenType::kTypeComma, i, i+1);
            this->_tokens.push_back(token);
            std::string token_comma = std::string(_statement.begin()+i, _statement.begin()+i+1);
            this->_token_strs.push_back(token_comma);
            i++;
        } else if (c=='(') {
            Token token(TokenType::kTypeLeftBracket, i, i+1);
            this->_tokens.push_back(token);
            std::string token_left_bracket = std::string(_statement.begin()+i, _statement.begin()+i+1);
            this->_token_strs.push_back(token_left_bracket);
            i++;
        } else if (c==')') {
            Token token(TokenType::kTypeRightBracket, i, i+1);
            this->_tokens.push_back(token);
            std::string token_right_bracket = std::string(_statement.begin()+i, _statement.begin()+i+1);
            this->_token_strs.push_back(token_right_bracket);
            i++;
        } else {
            std::cout << "[Error] Unknown illegal character: " << c << " at " << i << std::endl;
            return;
        }
    }
}

// 获得解析后的tokens
const std::vector<Token> &ExpressionOpParser::tokens() const {
    return this->_tokens;
}

// 获得token对应的str
const std::vector<std::string> &ExpressionOpParser::token_strs() const {
    return this->_token_strs;
}

// _Generate 根据Token构建对应的TokenNode，涉及建树
// 并且进行相应的语法检查
std::shared_ptr<TokenNode> ExpressionOpParser::_Generate(int32_t &index) {
    if (index >= this->_tokens.size()) {
        std::cout << "[Error] The index is over the _tokens.size." << std::endl;
    }
    const auto current_token = this->_tokens[index];
    // 或者用const auto current_token = this->_tokens.at(index); 可以不用边界判断
    if (current_token.token_type == TokenType::kTypeUnknown) {
        std::cout << "[Error] The token's type is unknown! The index is " << index << std::endl;
    }
    std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
    switch (current_token.token_type)
    {
    case TokenType::kTypeAdd : case TokenType::kTypeMul : {
        current_node->num_index = int(current_token.token_type);
        // 这里做了一些语法检查工作
        // 检查左括号
        index++;
        if (index>=this->_tokens.size() || 
            this->_tokens.at(index).token_type != TokenType::kTypeLeftBracket) {
            std::cout << "[Error] Missing left bracket!" << std::endl;
        }
        // 检查并生成左left token
        index++;
        if (index>=this->_tokens.size()) {
            std::cout << "[Error] Missing correspond left token!" << std::endl;
        }
        const auto left_token = this->_tokens.at(index);
        if (left_token.token_type == TokenType::kTypeData || 
            left_token.token_type == TokenType::kTypeAdd ||
            left_token.token_type == TokenType::kTypeMul) {
            // 生成left node
            current_node->left = _Generate(index);
        } else {
            std::cout << "[Error] Unknown left token type!" << std::endl;
        }
        // 检查逗号
        index++;
        if (index>=this->_tokens.size() || 
            this->_tokens.at(index).token_type != TokenType::kTypeComma) {
            std::cout << "[Error] Missing comma!" << std::endl;
        }
        // 检查并生成右right token
        index++;
        if (index>=this->_tokens.size()) {
            std::cout << "[Error] Missing correspond right token!" << std::endl;
        }
        const auto right_token = this->_tokens.at(index);
        if (right_token.token_type == TokenType::kTypeData || 
            right_token.token_type == TokenType::kTypeAdd ||
            right_token.token_type == TokenType::kTypeMul) {
            // 生成right node
            current_node->right = _Generate(index);
        } else {
            std::cout << "[Error] Unknown right token type!" << std::endl;
        }
        // 检查右括号
        index++;
        if (index>=this->_tokens.size() || 
            this->_tokens.at(index).token_type != TokenType::kTypeRightBracket) {
            std::cout << "[Error] Missing right bracket!" << std::endl;
        }
        break;
    }
    case TokenType::kTypeData : {
        // 跳过@
        uint32_t start_pos = current_token.start_pos + 1;
        uint32_t end_pos = current_token.end_pos;
        if (start_pos>=end_pos || end_pos > _statement.length()) {
            std::cout << "[Error] Current token has a wrong lenght!" << std::endl;
        }
        // 转为数字
        const std::string &str_number = std::string(
            _statement.begin()+start_pos,
            _statement.begin()+end_pos
        );
        current_node->num_index = std::stoi(str_number);
        current_node->left = nullptr;
        current_node->right = nullptr;
        break;
    }
    default: {
        std::cout << "[Error] The token's type is illegal! The index is " << index << std::endl;
        break;
    }

    }
    return current_node;
}

// 将语法树转化为逆波兰式(后序遍历)
void ReversePolish(const std::shared_ptr<TokenNode> &root,
                   std::vector<std::shared_ptr<TokenNode>> &reverse_polish) {
    if (root!=nullptr) {
        ReversePolish(root->left, reverse_polish);
        ReversePolish(root->right, reverse_polish);
        reverse_polish.push_back(root);
    }
}

// 生成语法树的入口，并返回语法树的逆波兰表达式的形式
std::vector<std::shared_ptr<TokenNode>> ExpressionOpParser::Generate() {
    if (this->_tokens.empty()) {
        this->Tokenizer(true);
    }
    int index = 0;
    std::shared_ptr<TokenNode> root = _Generate(index);
    if (root == nullptr || index!=_tokens.size()-1) {
        std::cout << "[Error] Generate token tree error!" << std::endl;
    }

    // 转换为逆波兰式
    std::vector<std::shared_ptr<TokenNode>> reverse_polish;
    ReversePolish(root, reverse_polish);
    return reverse_polish;
}


} // namespace lcnn
