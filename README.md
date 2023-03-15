# lcnn
lcnn for deep learning inference

## 项目用意

一直想从头写一个深度学习模型推理框架，但是迟迟没有动手，直到[KuiperInfer](https://github.com/zjhellofss/KuiperInfer)的出现再一次调动了我的兴趣！

但是当自己开始设计的时候，发现好像也不是那么简单，对于只进行过项目局部开发的我来说，从零设计有亿些难，于是准备从借鉴KuiperInfer开始，走上了一条仿写加局部重建的路（重复造轮子~），所以本项目主要用途是记录自己在推理框架实现方面的学习与应用！

目前与KuiperInfer在数据类上（计算图、算子节点等）基本一致，支持pnnx模型文件导入，不同点在于相关实现和工程组织上略有不同，争取求同存异慢慢发展出独特的亮点吧~

## 项目目标

- [x] 计算图可以正确跑通：实现Relu和ExprOp，构建简单计算图
- [ ] 支持Conv2d等常用算子，支持ResNet等网络模型推理
- [ ] 支持多硬件后端，像CUDA或者CPU下的其他数学计算库，如Eigen
- [ ] 或者更进一步，对硬件计算单元进行抽象，更高一层封装
- [ ] 支持onnx格式模型文件导入，主要想体验一下实现图优化pass（也是在造轮子~）
- [ ] 提供Python接口
- [ ] ......

## 致谢

非常感谢[KuiperInfer](https://github.com/zjhellofss/KuiperInfer)给我指引了方向！
