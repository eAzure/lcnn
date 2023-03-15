/**
 * graph.cc
 * 计算图
 * [by lgx 2023-3-1]
*/

#include "ir/graph.h"

namespace lcnn {

// 构造函数
Graph::Graph(std::string param_file_path, std::string bin_file_path) {
    this->_param_file_path = param_file_path;
    this->_bin_file_path = bin_file_path;
}

// 获取计算图对应的模型结构文件路径
const std::string& Graph::get_param_file_path() const {
    return this->_param_file_path;
}

// 获取计算图对应的权重文件路径
const std::string& Graph::get_bin_file_path() const {
    return this->_bin_file_path;
}

/********* 计算图初始化 *********/
/** InitOperatorInputOperands()
 * 初始化算子输入Operands
*/
void Graph::InitOperatorInputOperands(const std::vector<pnnx::Operand*>& pnnx_inputs,
                                      const std::shared_ptr<Operator>& con_operator) {
    for (const pnnx::Operand* pnnx_input : pnnx_inputs) {
        // 准备操作数
        const pnnx::Operator* pnnx_producer = pnnx_input->producer;
        std::shared_ptr<Operand> con_operand = std::make_shared<Operand>();
        con_operand->name = pnnx_producer->name;
        con_operand->shape = pnnx_input->shape;

        switch (pnnx_input->type)
        {
        case 1:
            con_operand->type = OperandDataType::kTypeFloat32;
            break;
        case 0:
            con_operand->type = OperandDataType::kTypeUnknown;
            break;
        default:
            std::cout << "[Error] Unknown pnnx_input dtype: " << pnnx_input->type;
            break;
        }
        con_operator->input_operands.insert({pnnx_producer->name, con_operand});
        con_operator->input_operands_seq.push_back(con_operand);
    }
}
/** InitOperatorOutputOperands()
 * 初始化算子输出Operand和输出Operators相关
*/
void Graph::InitOperatorOutputOperands(const std::vector<pnnx::Operand*>& pnnx_outputs,
                                       const std::shared_ptr<Operator>& con_operator) {
    // 目前只支持一个输出operand
    if (pnnx_outputs.size() > 1) {
        std::cout << "[Error] Currently operator support one output!" << std::endl;
        return;
    }
    const pnnx::Operand* pnnx_output = pnnx_outputs.at(0);
    if (!pnnx_output) {
        std::cout << "[Error] pnnx_output is nullptr!" << std::endl;
        return;
    }
    // 初始化输出Operand相关
    std::shared_ptr<Operand> con_output_operand = std::make_shared<Operand>();
    con_output_operand->name = pnnx_output->name + "_output";
    con_output_operand->shape = pnnx_output->shape;
    switch (pnnx_output->type)
    {
    case 1:
        con_output_operand->type = OperandDataType::kTypeFloat32;
        break;
    case 0:
        con_output_operand->type = OperandDataType::kTypeUnknown;
        break;
    default:
        std::cout << "[Error] Unknown pnnx_input dtype: " << pnnx_output->type;
        break;
    }
    con_operator->output_operand = con_output_operand;

    // 初始化输出Operators相关
    const auto& pnnx_consumers = pnnx_output->consumers;
    // std::cout << "operator name: " << con_operator->name << std::endl;
    for (const auto& pnnx_consumer : pnnx_consumers) {
        // std::cout << "output_name: " << pnnx_consumer->name << std::endl;
        con_operator->output_names.push_back(pnnx_consumer->name);
        // // 构建节点之间的关系
        // con_operator->output_operators.insert({pnnx_consumer->name, this->_name_operator_map[pnnx_consumer->name]});
        // std::cout << "hello: " << this->_name_operator_map[pnnx_consumer->name]->name << std::endl;
    }
    // std::cout << "----------------" << std::endl;
}
/** InitOperatorAttrs()
 * 初始化算子属性
*/
void Graph::InitOperatorAttrs(const std::map<std::string, pnnx::Attribute>& pnnx_attrs,
                              const std::shared_ptr<Operator>& con_operator) {
    for (const auto& pnnx_attr : pnnx_attrs) {
        const std::string& pnnx_attr_name = pnnx_attr.first;
        const pnnx::Attribute& pnnx_attr_attr = pnnx_attr.second;
        switch (pnnx_attr_attr.type)
        {
        case 1: {
            std::shared_ptr<OperatorAttribute> con_attribute = std::make_shared<OperatorAttribute>();
            con_attribute->type = OperandDataType::kTypeFloat32;
            con_attribute->shape = pnnx_attr_attr.shape;
            con_attribute->weight_data = pnnx_attr_attr.data;
            con_operator->attribute.insert({pnnx_attr_name, con_attribute});
            break;
        }
        default: {
            std::cout << "[Error] Unknown attribute type!" << std::endl;
            break;
        }
        }
    }
}
/**
 * InitOperatorParams()
 * 初始化算子参数
*/
void Graph::InitOperatorParams(const std::map<std::string, pnnx::Parameter>& pnnx_params,
                               const std::shared_ptr<Operator>& con_operator) {
    for (const auto& pnnx_param : pnnx_params) {
        const std::string& pnnx_param_name = pnnx_param.first;
        const pnnx::Parameter& pnnx_param_param = pnnx_param.second;
        const int pnnx_type = pnnx_param_param.type;
        switch (pnnx_type)
        {
        case int(OperatorParameterType::kTypeUnknown): {
            OperatorParameter* operator_param = new OperatorParameter;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeBool): {
            OperatorParameterBool* operator_param = new OperatorParameterBool;
            operator_param->value = pnnx_param_param.b;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeInt): {
            OperatorParameterInt* operator_param = new OperatorParameterInt;
            operator_param->value = pnnx_param_param.i;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeFloat): {
            OperatorParameterFloat* operator_param = new OperatorParameterFloat;
            operator_param->value = pnnx_param_param.f;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeString): {
            OperatorParameterString* operator_param = new OperatorParameterString;
            operator_param->value = pnnx_param_param.s;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeIntArray): {
            OperatorParameterIntArray* operator_param = new OperatorParameterIntArray;
            operator_param->value = pnnx_param_param.ai;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeFloatArray): {
            OperatorParameterFloatArray* operator_param = new OperatorParameterFloatArray;
            operator_param->value = pnnx_param_param.af;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        case int(OperatorParameterType::kTypeStringArray): {
            OperatorParameterStringArray* operator_param = new OperatorParameterStringArray;
            operator_param->value = pnnx_param_param.as;
            con_operator->params.insert({pnnx_param_name, operator_param});
            break;
        }
        default: {
            std::cout << "[Error] Unknown parameter type!" << std::endl;
        }
        }
    }
}

/** Init()
 * 获取pnnx::Graph，从中更新Operators列表
*/
bool Graph::Init() {
    if (this->_bin_file_path.empty() || this->_param_file_path.empty()) {
        std::cout << "[Error] The bin path or param path is empty!" << std::endl;
        return false;
    }
    // 从pnnx 模型文件中导入计算图
    this->_graph = std::make_unique<pnnx::Graph>();
    int load_result = this->_graph->load(_param_file_path, _bin_file_path);
    if (load_result != 0) {
        std::cout << "[Error] Load pnnx graph error!" << std::endl;
        return false;
    }
    // 从图中获得operator列表
    std::vector<pnnx::Operator*> operators = this->_graph->ops;
    if (operators.empty()) {
        std::cout << "[Error] Load graph->ops error!" << std::endl;
        return false;
    }
    this->_operators.clear();
    // 遍历初始化自己的Operator
    for (const pnnx::Operator* pnnx_operator : operators) {
        if (!pnnx_operator) {
            std::cout << "[Error] Meet the empty pnnx_operator node!" << std::endl;
            return false;
        } else {
            std::shared_ptr<Operator> con_operator = std::make_shared<Operator>();
            // 初始化Operator name和type
            con_operator->name = pnnx_operator->name;
            // 记录Operator名称与Operator节点的对应关系
            this->_name_operator_map.insert({con_operator->name, con_operator});
            // std::cout << "insert: " << this->_name_operator_map[con_operator->name]->name << std::endl;
            con_operator->type = pnnx_operator->type;
            // 初始化Input Operand
            const std::vector<pnnx::Operand*>& pnnx_inputs = pnnx_operator->inputs;
            if (!pnnx_inputs.empty()) {
                InitOperatorInputOperands(pnnx_inputs, con_operator);
            }
            // 初始化Output Operand
            const std::vector<pnnx::Operand*>& pnnx_outputs = pnnx_operator->outputs;
            if (!pnnx_outputs.empty()) {
                InitOperatorOutputOperands(pnnx_outputs, con_operator);
            }
            // 初始化attribute
            const std::map<std::string, pnnx::Attribute>& pnnx_attrs = pnnx_operator->attrs;
            if (!pnnx_attrs.empty()) {
                InitOperatorAttrs(pnnx_attrs, con_operator);
            }
            // 初始化parameter
            const std::map<std::string, pnnx::Parameter>& pnnx_params = pnnx_operator->params;
            if (!pnnx_params.empty()) {
                InitOperatorParams(pnnx_params, con_operator);
            }
            // 初始化完成
            this->_operators.push_back(con_operator);
        }
    }
    // 更新计算图状态
    this->_graph_state = GraphState::NeedBuild;
    return true;
}

/********** InitOperatorInputAndOutput ***********/
/** InitOperatorInput()
 * input_datas.resize(batch_size)
*/
void Graph::InitOperatorInput(const std::vector<std::shared_ptr<Operator>>& con_operators) {
    if (con_operators.empty()) {
        std::cout << "[Error] Con_operators for init input shape is empty!" << std::endl;
        return;
    }
    for (const auto& con_operator : con_operators) {
        if (con_operator->input_operands.empty()) {
            continue;
        } else {
            const std::map<std::string, std::shared_ptr<Operand>>& input_operands = con_operator->input_operands;
            for (const auto& input_operand_iter : input_operands) {
                const auto& input_operand = input_operand_iter.second;
                const auto& type = input_operand->type;
                if (type != OperandDataType::kTypeFloat32) {
                    std::cout << "[Error] The graph only support float32 now!" << std::endl;
                    return;
                }
                const auto& input_operand_shape = input_operand->shape;
                // 获取数据空间
                auto& input_datas = input_operand->datas;
                // 一些检查
                // 1. 首先检查input_operand_shape是否为空
                if (input_operand_shape.empty()) {
                    std::cout << "[Error] The input_operand_shape is empty!" << std::endl;
                    return;
                }
                // 获取batch_size大小
                const int32_t batch_size = input_operand_shape.at(0);
                // 2. 检查batch_size大小
                if (batch_size <= 0) {
                    std::cout << "[Error] The batch size is negative!" << std::endl;
                    return;
                }
                // 3. 检查shape尺寸
                if (input_operand_shape.size()<2&&input_operand_shape.size()>4) {
                    std::cout << "[Error] Unsupported tensor shape size: " << input_operand_shape.size() << std::endl;
                    return;
                }
                // 4. 检查input_datas是否初始化过
                if (!input_datas.empty()) {
                    if (input_datas.size() != batch_size) {
                        std::cout << "[Error] The input_data.size is not equal to batch_size!" << std::endl;
                        return;
                    }
                } else {
                    input_datas.resize(batch_size);
                }
            }
        }
    }
}
/** InitOperatorOutput()
 * 这里和上面保持一致即可
*/
void Graph::InitOperatorOutput(const std::vector<std::shared_ptr<Operator>>& con_operators) {
    if (con_operators.empty()) {
        std::cout << "[Error] Con_operators for init output shape is empty!" << std::endl;
        return;
    }
    for (const auto& con_operator : con_operators) {
        if (con_operator->output_operand == nullptr) continue;
        auto& output_operand = con_operator->output_operand;
        if (output_operand->shape.empty()) {
            std::cout << "[Error] output_operans shape is empty!" << std::endl;
            return;
        }
        const int32_t batch_size = output_operand->shape.at(0);
        if (batch_size <= 0) {
            std::cout << "[Error] The batch size is negative!" << std::endl;
            return;
        }
        if (output_operand->shape.size()<2 && output_operand->shape.size()>4) {
            std::cout << "[Error] Unsupported tensor shape size: " << output_operand->shape.size() << std::endl;
            return;
        }
        // 获取数据空间
        auto& output_datas = output_operand->datas;
        if (!output_datas.empty()) {
            if (output_datas.size() != batch_size) {
                std::cout << "[Error] The output_data.size is not equal to batch_size!" << std::endl;
                return;
            }
        } else {
            output_datas.resize(batch_size);            
        }
    }
}

/*********** Build ***********/

// 生成算子拓扑序的逆序
/**
 * DFS的逆序即为算子拓扑序，所以这里是采用DFS遍历计算图中Operator的节点
*/
void Graph::ReverseTopo(const std::shared_ptr<Operator>& current_operator) {
    current_operator->has_forward = true;
    const auto& next_operator_maps = current_operator->output_operators;
    for (const auto& next_operator_map : next_operator_maps) {
        const auto& next_operator = next_operator_map.second;
        if (next_operator != nullptr) {
            if (!next_operator->has_forward) {
                this->ReverseTopo(next_operator);
            }
        }
    }
    this->_topo_operators.push_back(current_operator);
}

// 构建计算图
/**
 * 根据结构文件和权重文件初始化计算图，需要遍历pnnx
*/
void Graph::Build(const std::string& input_name, const std::string& output_name) {
    // 首先需要遍历pnnx图结构获取对应的算子集合，然后构建算子间的关系
    if (this->_graph_state == GraphState::NeedInit) {
        bool init_graph_state = Init();
        if (!init_graph_state) {
            std::cout << "[Error] Init graph error!" << std::endl;
            return;
        }
    }

    if (this->_graph_state < GraphState::NeedBuild) {
        std::cout << "[Error] Graph status error, current state is " << int(_graph_state) << std::endl;
        return;
    }

    if (this->_operators.empty()) {
        std::cout << "[Error] Operators is empty! Init Error! " << std::endl;
        return;
    }

    if (this->_graph_state == GraphState::Complete) {
        return;
    }

    // 构建图关系
    for (const auto& current_operator : this->_operators) {
        const std::vector<std::string>& output_operator_names = current_operator->output_names;
        for (const std::string output_operator_name : output_operator_names) {
            current_operator->output_operators.insert({output_operator_name, this->_name_operator_map[output_operator_name]});
        }
    }

    this->_input_operators.clear();
    this->_output_operators.clear();
    // 创建对应的Op
    for (const auto& con_operator : this->_operators) {
        if (con_operator->type == "pnnx.Input") {
            this->_input_operators.insert({con_operator->name, con_operator});
        } else if (con_operator->type == "pnnx.Output") {
            this->_output_operators.insert({con_operator->name, con_operator});
        } else {
            std::shared_ptr<Op> op = OpRegisterer::CreateOp(con_operator);
            if (op==nullptr) {
                std::cout << "[Error] Op create error!" << std::endl;
                return;
            }
            con_operator->op = op;
            op->set_operator(con_operator);
        }
    }
    // 设置输入输出节点名称
    this->_input_name = input_name;
    this->_output_name = output_name;

    /*** 准备算子输入、输出数据空间 ***/
    // 加速数据准备时间，第一次加载，第二次只负责检查可以加速
    // 准备算子输入
    InitOperatorInput(this->_operators);
    // 准备算子输出
    InitOperatorOutput(this->_operators);

    // 构建算子拓扑序
    this->_topo_operators.clear();
    for (const auto& input_operator_map : _input_operators) {
        this->ReverseTopo(input_operator_map.second);
    }
    // 逆序构建算子拓扑序
    std::reverse(this->_topo_operators.begin(), this->_topo_operators.end());

    // 图构建完成
    this->_graph_state = GraphState::Complete;
    // 将_graph置为空指针，_graph的作用在于作为pnnx::Graph的替身
    if (this->_graph != nullptr) {
        this->_graph.reset();
        this->_graph = nullptr;
    }

}

/********* Forward *********/

// 由执行完的节点准备后续节点的输入
void Graph::PrepareNextOperator(
        const std::shared_ptr<Operator>& current_operator,
        const std::vector<std::shared_ptr<Tensor<float>>>& current_operator_output_datas) {
    // 获得后续节点
    const auto& next_operator_maps = current_operator->output_operators;
    for (const auto& next_operator_map : next_operator_maps) {
        // 得到后续节点
        const auto& next_operator = next_operator_map.second;
        // 获得后续节点的输入操作数
        const auto& next_operator_input_operand_maps = next_operator->input_operands;
        if (next_operator_input_operand_maps.find(current_operator->name) !=
            next_operator_input_operand_maps.end()) {
            std::vector<std::shared_ptr<Tensor<float>>>& next_operator_input_datas = next_operator_input_operand_maps.at(current_operator->name)->datas;
            for (int i=0;i<next_operator_input_datas.size();i++) {
                next_operator_input_datas.at(i) = current_operator_output_datas.at(i);
            }
        }
    }
}

// 执行计算图 Forward
std::vector<std::shared_ptr<Tensor<float>>> Graph::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs) {
    if (this->_graph_state < GraphState::Complete) {
        std::cout << "[Error] Graph need be build!" << std::endl;
        return {};
    }
    if (this->_topo_operators.size() != this->_operators.size()) {
        std::cout << "[Error] Topo wrong!" << std::endl;
        return {};
    }

    for (const auto& con_operator : this->_topo_operators) {
        con_operator->has_forward = false;
    }

    for (const auto& current_operator : this->_topo_operators) {
        if (this->_input_operators.find(current_operator->name) !=
            this->_input_operators.end()) {
            current_operator->has_forward = true;
            PrepareNextOperator(current_operator, inputs);
        } else if (this->_output_operators.find(current_operator->name) !=
                   this->_output_operators.end()) {
            current_operator->has_forward = true;
            // 输出节点的输入就是输出
            current_operator->output_operand = current_operator->input_operands_seq.front();
        } else {
            InferStatus status = current_operator->op->Forward();
            if (status != InferStatus::kInferSuccess) {
                std::cout << "[Error] Infer failed: "<< int(status) << "!" << std::endl;
                return {};
            }
            current_operator->has_forward = true;
            PrepareNextOperator(current_operator, current_operator->output_operand->datas);
        }
    }
    // 检查是否所有节点都执行过了
    for (const auto& con_operator : this->_topo_operators) {
        if (!con_operator->has_forward) {
            std::cout << "[Error] " << con_operator->name << " has not been forward yet!" << std::endl;
            return {};
        }
    }
    // 输出节点中的输出值作为计算图的输出
    if (this->_output_operators.find(this->_output_name) != this->_output_operators.end()) {
        const auto& output_operator = this->_output_operators.at(this->_output_name);
        if (output_operator->output_operand == nullptr) {
            std::cout << "[Error] Output from " << this->_output_name << " is empty!" << std::endl;
            return {};
        }
        // 返回计算图结果
        return output_operator->output_operand->datas;
    }

    std::cout << "[Error] Can not find the output operator: " << this->_output_name << std::endl;
    return {};
}


} // namespace lcnn