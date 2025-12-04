# 更新日志

## [0.1.0] - 2025-12-04

### 新增
- 初始项目架构
- 视觉编码器（ResNet-18 with Spatial Softmax）
- 语言编码器（CLIP-based）
- 1D U-Net 扩散模型架构
- FiLM 和 Cross-Attention 条件注入机制
- 完整的 Diffusion Policy 实现
- 数据集加载和预处理（支持 Zarr/HDF5）
- 脚本专家系统（Pick, Push, Stack）
- 训练脚本（支持 WandB 日志）
- 评估脚本（带滚动时域控制）
- 数据采集脚本
- 可视化工具
- 完整文档（README, USAGE, ARCHITECTURE）
- 单元测试套件
- Jupyter notebook 示例

### 支持的任务
- PickCube: 单物体抓取
- PushCube: 平面推动
- StackCube: 物体堆叠

### 特性
- Action Chunking (预测16步)
- Receding Horizon Control (执行8步)
- DDPM 训练 + DDIM 推理加速
- 动作归一化和数据增强
- 多模态条件融合
- 完整的训练和评估流程

## 待实现
- [ ] Vision Transformer (ViT) 编码器
- [ ] 多相机支持
- [ ] 深度图和点云观测
- [ ] 强化学习微调
- [ ] 预训练模型权重
- [ ] Docker 容器化
- [ ] 更多任务支持
