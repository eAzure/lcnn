/**
 * expression.h
 * expression op
 * [by lgx 2023-02-26]
*/

#ifndef _LCNN_IR_OPS_EXPRESSION_H
#define _LCNN_IR_OPS_EXPRESSION_H

#include "ir/op/op.h"
#include "ir/operator.h"
#include "ir/op/expression_op_parse.h"

namespace lcnn {

class ExpressionOp:public Op {
public:
    ExpressionOp(const std::string &statement) 
        : Op("Expression"), _parser(std::make_unique<ExpressionOpParser>(statement)) {}
    // Forward
    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
    // GetInstance
    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<Operator> &con_operator,
                                                std::shared_ptr<Op> &expression_op);

private:
    // 每一个expression_op都对应着一个parser
    std::unique_ptr<ExpressionOpParser> _parser;
};

} // namespace lcnn

#endif // _LCNN_IR_OPS_EXPRESSION_H
