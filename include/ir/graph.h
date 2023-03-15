/**
 * graph.h
 * 计算图类
 * [by lgx 2023-3-1]
*/

#ifndef _LCNN_IR_GRAPH_H
#define _LCNN_IR_GRAPH_H

#include "pnnx_ir/ir.h"
#include "ir/operator.h"
#include "ir/op/op_register.h"
#include <map>
#include <vector>
#include <deque>

namespace lcnn {

enum class GraphState {
    NeedInit = -2, // 需要初始化
    NeedBuild = -1, // 需要build
    Complete = 0 // build完成
};

class Graph {
public:
    /**
     * 构造函数
     * @param param_file_path : 对应的模型结构文件路径
     * @param bin_file_path : 对应的权重文件路径
    */
    Graph(std::string param_file_path, std::string bin_file_path);

    /**
     * 获取计算图对应的模型结构文件路径
    */
    const std::string& get_param_file_path() const;
    
    /**
     * 获取计算图对应的权重文件路径
    */
    const std::string& get_bin_file_path() const;

    /**
     * 构建计算图
     * @param input_name : 输入节点名称
     * @param output_name : 输出节点名称
    */
    void Build(const std::string& input_name, const std::string& output_name);

    /**
     * 执行计算图
     * @param inputs : 输入数据
     * @return 计算得到的张量数据
    */
    std::vector<std::shared_ptr<Tensor<float>>> Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs
    );

private:
    // 计算图初始化，用于遍历pnnx获取节点信息
    bool Init();
    // Init中用到的一些初始化函数，包括初始化输入、输出Operand、Attrs、Params
    /**
     * 初始化算子输入Operand
     * @param pnnx_inputs : 从pnnx中获得的算子的输入
     * @param con_operator : 待初始化的算子
    */
    void InitOperatorInputOperands(const std::vector<pnnx::Operand*>& pnnx_inputs,
                                   const std::shared_ptr<Operator>& con_operator);
    /**
     * 初始化算子输出Operand
     * @param pnnx_outputs : 从pnnx中获得的算子的输出
     * @param con_operator : 待初始化的算子
    */
    void InitOperatorOutputOperands(const std::vector<pnnx::Operand*>& pnnx_outputs,
                                    const std::shared_ptr<Operator>& con_operator);
    /**
     * 初始化算子Attrs
     * @param pnnx_attrs : 从pnnx中获得的算子属性
     * @param con_operator : 待初始化的算子
    */
    void InitOperatorAttrs(const std::map<std::string, pnnx::Attribute>& pnnx_attrs,
                           const std::shared_ptr<Operator>& con_operator);
    /**
     * 初始化算子Params
     * @param pnnx_params : 从pnnx中获得的算子参数
     * @param con_operator : 待初始化的算子
    */
    void InitOperatorParams(const std::map<std::string, pnnx::Parameter>& pnnx_params,
                            const std::shared_ptr<Operator>& con_operator);

    // 初始化图中算子节点的输入输出，方便加速
    // 这里只对input_operand里面的data resize(batch)
    void InitOperatorInput(const std::vector<std::shared_ptr<Operator>>& con_operators);
    void InitOperatorOutput(const std::vector<std::shared_ptr<Operator>>& con_operators);


    /**
     * 判断后续节点是否准备好 visit_num is equal to input_operands.size()
     * @param con_operator 待判断的operator
    */
    bool CheckOperatorReady(const std::shared_ptr<Operator>& con_operator);

    /**
     * 由当前Operator执行完成初始化后续节点，并将后续节点插入到执行队列中
     * @param current_operator : 当前Operator
     * @param run_operator_queue : Operator执行队列
     * @param operator_output_data : 当前Operator输出，后续节点的输入
    */
    void PrepareNextOperator(
        const std::shared_ptr<Operator>& current_operator,
        const std::vector<std::shared_ptr<Tensor<float>>>& operator_output_datas
    );

    /**
     * 由输入节点出发，生成算子拓扑序
     * @param current_operator : 出发节点
    */
    void ReverseTopo(const std::shared_ptr<Operator>& current_operator);


    GraphState _graph_state = GraphState::NeedInit; // 定义当前图状态
    std::unique_ptr<pnnx::Graph> _graph; // 对应的pnnx Graph
    std::string _input_name; // 图输入节点名称
    std::string _output_name; // 图输出节点名称
    std::string _param_file_path; // 对应的模型结构文件路径
    std::string _bin_file_path; // 对应的权重文件路径
    std::map<std::string, std::shared_ptr<Operator>> _input_operators; // 对应的输入节点
    std::map<std::string, std::shared_ptr<Operator>> _output_operators; // 对应的输出节点
    std::vector<std::shared_ptr<Operator>> _operators; // 图中计算节点序列
    std::vector<std::shared_ptr<Operator>> _topo_operators; // 图中计算节点拓扑序
    std::map<std::string, std::shared_ptr<Operator>> _name_operator_map; // 图中计算节点 名称与Op对应关系 // TODO(new add)
};

} // namespace lcnn

#endif // _LCNN_IR_GRAPH_H