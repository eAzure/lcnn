/**
 * expression_op_parse.h
 * 解析expression_op
 * 解析器类
 * expression_op示例：add(add(add(@0,@1),@2),add(@3,@4))
 * [by lgx 2023-02-26]
*/

#ifndef _LCNN_IR_OP_EXPRESSION_PARSE_H
#define _LCNN_IR_OP_EXPRESSION_PARSE_H

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm> // std::remove_if

namespace lcnn {

// TokenType
// 这里采用负数的原因是TokenNode中的num_index是记录这个的
// 而Data记录的是@后面的数字，其为正数，所以为了区分，这里采用负数
enum class TokenType {
    kTypeUnknown = -2,
    kTypeData = -3,
    kTypeComma = -4,
    kTypeAdd = -5,
    kTypeMul = -6,
    kTypeLeftBracket = -7,
    kTypeRightBracket = -8
};

// Token
struct Token
{
    TokenType token_type = TokenType::kTypeUnknown;
    int32_t start_pos = 0; // token起始pos
    int32_t end_pos = 0; // token终止pos

    Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
        : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {}
    Token() = default;
};

// TokenNode 用于构造语法树
// 语法树中只存在Data、Add和Mul Token，不包括"(", ")", ","
struct TokenNode
{   
    // 对于DataType就是@后面的数字，Add和Mul则为(int)TokenType
    int32_t num_index = -1;
    std::shared_ptr<TokenNode> left = nullptr; // 左节点
    std::shared_ptr<TokenNode> right = nullptr; // 右节点
    
    TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right)
        : num_index(num_index), left(std::move(left)), right(std::move(right)) {}
    TokenNode() = default;
};

// ExpressionOpParser
// 解析器类
class ExpressionOpParser {
public:
    explicit ExpressionOpParser(std::string statement) : _statement(std::move(statement)) {}
    // 词法分析
    /**
     * re_tokenize: bool 是否重新进行词法分析
    */
    void Tokenizer(bool re_tokenize = false);
    // 生成语法树的入口，并返回语法树的逆波兰表达式的形式
    std::vector<std::shared_ptr<TokenNode>> Generate();
    // 获得词法分析结果
    const std::vector<Token> &tokens() const;
    // 获取_token_strs
    const std::vector<std::string> &token_strs() const;
private:
    // _Generate 根据Token构建对应的TokenNode，涉及建树
    // 并且进行相应的语法检查，为Generate的内部调用接口
    std::shared_ptr<TokenNode> _Generate(int32_t &index);
    // _token序列，由词法分析获得
    std::vector<Token> _tokens;
    // token对应的str表达，如AddToken对应"add"
    std::vector<std::string> _token_strs;
    // 对应的表达式，如"@1"
    std::string _statement;
};


} // namespace lcnn

#endif // _LCNN_IR_OP_EXPRESSION_PARSE_H
