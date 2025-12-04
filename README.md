# LCDP-Sim: Language-Conditioned Diffusion Policy

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LCDP-Sim 是一个端到端视觉-语言-动作 (VLA) 系统，利用扩散策略 (Diffusion Policy) 通过自然语言指令控制机械臂。

## 🚀 核心特性

- **端到端学习**: RGB 图像 + 语言指令 -> 动作序列
- **扩散策略**: 基于 DDPM/DDIM 的动作生成
- **语言条件**: 集成 CLIP 文本编码器
- **动作分块**: 预测未来 16 步轨迹 (Action Chunking)

## �� 项目结构

```
LCDP-Sim/
├── configs/                    # 配置文件 (环境, 训练)
├── lcdp/                       # 核心代码包
│   ├── models/                 # 模型定义 (Diffusion, U-Net, Encoders)
│   ├── data/                   # 数据处理 (Dataset, Loader)
│   └── envs/                   # 环境包装器
├── scripts/                    # 可执行脚本
│   ├── train.py                # 训练入口
│   ├── eval.py                 # 评估与可视化
│   ├── collect_data.py         # 仿真数据采集
│   ├── convert_real_data.py    # 真实数据转换工具
│   └── visualize.py            # 数据可视化工具
├── checkpoints/                # 模型权重
├── data/                       # 数据集 (.zarr)
├── logs/                       # 训练日志
├── videos/                     # 评估视频
└── requirements.txt            # 依赖包
```

## 🛠️ 安装

```bash
# 创建环境
conda create -n lcdp python=3.8 -y
conda activate lcdp

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

## 🎮 使用指南

### 1. 数据准备

**方式 A: 使用现有数据集 (推荐)**
自动下载并处理 ManiSkill2 官方演示数据：
```bash
python scripts/prepare_maniskill_data.py --env PickCube-v0 --output data/pick_cube_demos.zarr
```

**方式 B: 仿真数据采集**
使用脚本专家在本地生成数据：
```bash
python scripts/collect_data.py --env PickCube-v0 --num-episodes 100 --output data/pick_cube_demos.zarr
```

**方式 C: 真实数据**
将真实数据整理为图片和动作序列，使用转换脚本：
```bash
python scripts/convert_real_data.py --input_dir /path/to/raw_data --output data/real_robot_data.zarr
```

### 2. 训练

```bash
python scripts/train.py --config configs/train_config.yaml --data data/pick_cube_demos.zarr --output checkpoints/model_v1
```

*注意: 8GB 显存用户建议在 `configs/train_config.yaml` 中将 `batch_size` 调小至 16 或 8。*

### 3. 评估与可视化

**仿真评估 (保存视频):**
```bash
python scripts/eval.py --checkpoint checkpoints/model_v1/best.pth --env PickCube-v0 --save-videos
```

**实时可视化 (Human Render):**
```bash
python scripts/eval.py --checkpoint checkpoints/model_v1/best.pth --env PickCube-v0 --render
```

## 📄 许可证

MIT License
