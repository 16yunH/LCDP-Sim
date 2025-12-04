# LCDP-Sim 项目架构说明

## 核心设计理念

LCDP-Sim 采用模块化设计，将复杂的机器人学习系统分解为独立、可复用的组件。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Data Collection → Dataset → DataLoader → Model → Training  │
│       ↓              ↓          ↓           ↓         ↓      │
│  Scripted Expert  Zarr/H5   PyTorch   Diffusion   Optimizer │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Model Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐      ┌──────────┐                            │
│  │  Image   │      │Language  │                            │
│  │  (RGB)   │      │Instruction│                            │
│  └────┬─────┘      └────┬─────┘                            │
│       │                 │                                    │
│       ↓                 ↓                                    │
│  ┌──────────┐      ┌──────────┐                            │
│  │ ResNet/  │      │  CLIP    │                            │
│  │   ViT    │      │  Text    │                            │
│  │ Encoder  │      │ Encoder  │                            │
│  └────┬─────┘      └────┬─────┘                            │
│       │                 │                                    │
│       │   ┌─────────────┘                                   │
│       │   │                                                  │
│       ↓   ↓                                                  │
│  ┌──────────────┐                                           │
│  │ Conditioning  │  (FiLM / Cross-Attention)                │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ↓                                                    │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Noisy Actions│ + t →│   U-Net1D    │→ Predicted Noise   │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
│  Iterative Denoising → Clean Action Sequence                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. 数据流水线 (`lcdp/data/`)

**dataset.py**
- `RobotDataset`: 加载和预处理演示数据
- 支持 Zarr 和 HDF5 格式
- 滑动窗口采样，生成 (obs, action_sequence) 对
- 动作归一化和数据增强

**关键设计**：
- 使用 `action_horizon` 实现 Action Chunking
- 自动计算动作统计量用于归一化
- 支持多种图像预处理策略

### 2. 模型组件 (`lcdp/models/`)

**vision_encoder.py**
- `VisionEncoder`: 视觉特征提取
- 支持 ResNet-18 和 ViT (可扩展)
- `SpatialSoftmax`: 保留空间信息的池化层
- 可选冻结预训练权重

**language_encoder.py**
- `LanguageEncoder`: CLIP 文本编码器
- 默认冻结以利用预训练知识
- 可配置输出维度

**unet1d.py**
- `UNet1D`: 1D U-Net 用于时序动作去噪
- `SinusoidalPositionEmbedding`: 时间步编码
- `ResidualBlock1D`: 带时间条件的残差块
- 下采样/上采样路径 + 跳跃连接

**conditioning.py**
- `FiLM`: 轻量级特征调制
- `CrossAttention`: 强表达力的注意力机制
- `MultimodalFiLM/CrossAttention`: 融合视觉和语言

**diffusion_policy.py**
- `DiffusionPolicy`: 完整的策略网络
- 集成所有组件
- 训练时：计算扩散损失
- 推理时：DDIM 采样生成动作

### 3. 环境与专家 (`lcdp/envs/`)

**scripted_expert.py**
- `ScriptedExpert`: 基于规则的专家策略
- 状态机设计：approach → grasp → lift → move → release
- 支持 pick, push, stack 任务
- 添加噪声以提高鲁棒性

### 4. 训练和评估 (`scripts/`)

**train.py**
- `Trainer`: 管理完整训练流程
- 支持 WandB 日志记录
- 自动保存检查点
- 学习率调度

**eval.py**
- `Evaluator`: 在仿真环境中测试策略
- 滚动时域控制 (Receding Horizon)
- 统计成功率、奖励等指标
- 保存评估视频

**collect_data.py**
- 使用脚本专家收集演示
- 支持多任务、多场景
- 自动过滤失败轨迹
- 保存为 Zarr/HDF5

**visualize.py**
- 可视化动作序列
- 可视化去噪过程
- 对比不同指令的预测

## 配置系统 (`configs/`)

使用 YAML 配置文件管理超参数：

**train_config.yaml**
- 模型架构参数
- 训练超参数
- 数据加载设置
- 日志和检查点配置

**env_config.yaml**
- 仿真环境设置
- 机器人配置
- 任务定义
- 数据收集参数

## 关键设计决策

### 1. Action Chunking
预测未来多步而非单步，优点：
- 轨迹更平滑
- 减少累积误差
- 利用扩散模型的序列生成能力

### 2. Receding Horizon Control
只执行预测序列的前几步，优点：
- 实时应对扰动
- 利用最新观测
- 提高鲁棒性

### 3. 条件注入机制
支持两种融合方式：
- **FiLM**: 参数量小，推理快
- **Cross-Attention**: 表达力强，性能更好

### 4. 扩散策略
训练用 DDPM (100步)，推理用 DDIM (10步)：
- 训练稳定
- 推理加速 10x
- 性能损失 < 3%

## 扩展性考虑

### 添加新的视觉编码器
在 `vision_encoder.py` 中添加新的编码器类：
```python
class NewEncoder(nn.Module):
    def __init__(self, output_dim):
        # Implementation
        pass
```

### 添加新的条件机制
在 `conditioning.py` 中实现新的融合模块。

### 支持新的任务
1. 在 `configs/env_config.yaml` 中定义任务
2. 在 `scripted_expert.py` 中添加专家策略
3. 收集数据并训练

## 性能优化建议

1. **数据加载**: 使用 `num_workers > 0` 并行加载
2. **混合精度**: 使用 `torch.cuda.amp` 加速训练
3. **梯度累积**: 在小 GPU 上模拟大 batch size
4. **模型剪枝**: 减少 U-Net 通道数用于边缘部署

## 测试策略

- 单元测试 (`tests/test_*.py`)
- 集成测试（端到端流水线）
- 性能基准测试
- 消融实验验证设计选择

## 文档结构

```
docs/
├── USAGE.md              # 使用指南
├── ARCHITECTURE.md       # 本文件
├── API.md               # API 文档（待添加）
└── EXPERIMENTS.md       # 实验结果（待添加）
```

## 未来工作

- [ ] 添加 Vision Transformer 支持
- [ ] 实现多模态观测（深度图、点云）
- [ ] 支持强化学习微调
- [ ] Sim-to-Real 迁移实验
- [ ] 分布式训练支持
- [ ] 模型压缩和量化
