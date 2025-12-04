# LCDP-Sim 使用指南

## 快速开始

### 1. 环境安装

**Windows 用户（自动安装）：**
```powershell
# 克隆仓库
git clone https://github.com/yourusername/LCDP-Sim.git
cd LCDP-Sim

# 运行自动安装脚本（会创建 lcdp conda 环境）
.\setup.ps1
```

**手动安装（所有平台）：**
```bash
# 克隆仓库
git clone https://github.com/yourusername/LCDP-Sim.git
cd LCDP-Sim

# 创建 conda 环境
conda create -n lcdp python=3.8 -y
conda activate lcdp

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

**后续使用时：**
每次使用前需要激活环境：
```bash
conda activate lcdp
```

### 2. 数据采集

使用脚本专家在仿真环境中收集演示数据：

```bash
# 收集 PickCube 任务的演示
python scripts/collect_data.py \
    --env PickCube-v0 \
    --num-episodes 100 \
    --output data/pick_cube_demos.zarr \
    --save-videos \
    --video-dir data/videos/pick_cube

# 收集 PushCube 任务的演示
python scripts/collect_data.py \
    --env PushCube-v0 \
    --num-episodes 100 \
    --output data/push_cube_demos.zarr

# 收集 StackCube 任务的演示
python scripts/collect_data.py \
    --env StackCube-v0 \
    --num-episodes 100 \
    --output data/stack_cube_demos.zarr
```

### 3. 训练模型

```bash
# 使用默认配置训练
python scripts/train.py \
    --config configs/train_config.yaml \
    --data data/pick_cube_demos.zarr \
    --output checkpoints/pick_cube_model

# 使用 WandB 进行日志记录
# 首先登录: wandb login
python scripts/train.py \
    --config configs/train_config.yaml \
    --data data/pick_cube_demos.zarr
```

**训练配置说明：**
- 修改 `configs/train_config.yaml` 来调整超参数
- 默认使用 Cross-Attention 作为条件注入机制
- 支持 FiLM 和 Cross-Attention 两种融合方式
- 使用 DDPM (100步) 进行训练，DDIM (10步) 进行推理加速

### 4. 评估模型

```bash
# 在仿真环境中评估
python scripts/eval.py \
    --checkpoint checkpoints/pick_cube_model/best.pth \
    --env PickCube-v0 \
    --instruction "Pick up the red cube" \
    --num-episodes 50 \
    --save-videos \
    --video-dir videos/evaluation

# 可视化单个轨迹
python scripts/visualize.py \
    --checkpoint checkpoints/pick_cube_model/best.pth \
    --instruction "Pick up the blue cube and place it on the left"
```

## 进阶使用

### 自定义模型配置

修改 `configs/train_config.yaml`:

```yaml
model:
  # 切换视觉编码器
  vision_encoder: "resnet18"  # 或 "vit"
  
  # 切换条件注入机制
  conditioning_type: "cross_attention"  # 或 "film"
  
  # 调整 U-Net 规模
  unet_base_channels: 256
  unet_channel_mult: [1, 2, 4]  # 增加通道数以提升容量
  
  # 扩散步数
  num_diffusion_steps: 100
```

### 数据增强

在 `configs/train_config.yaml` 中启用数据增强：

```yaml
dataset:
  augment: true  # 启用颜色抖动等增强
```

### 多任务训练

收集多个任务的数据并合并：

```python
import zarr

# 合并数据集
root = zarr.open('data/multi_task_demos.zarr', mode='w')
# ... 自定义合并逻辑
```

### 滚动时域控制 (Receding Horizon Control)

在推理时，模型预测未来 `action_horizon` 步（如16步），但只执行前 `execute_horizon` 步（如8步）：

```python
# 在 eval.py 中
action_sequence = model.get_action(
    images=image,
    instructions=[instruction],
    num_inference_steps=10  # DDIM 步数
)

# 只执行前 8 步
actions_to_execute = action_sequence[0, :, :8]
```

这种策略能够：
- 实时应对环境扰动
- 利用最新观测重新规划
- 提高长程任务的成功率

## 实验建议

### 对比实验

1. **条件注入机制对比**
   ```bash
   # FiLM
   python scripts/train.py --config configs/film_config.yaml
   
   # Cross-Attention
   python scripts/train.py --config configs/cross_attn_config.yaml
   ```

2. **Action Chunking 消融实验**
   - 预测 1 步 vs. 16 步
   - 观察长程任务的性能差异

3. **采样步数对比**
   - DDPM 100步 vs. DDIM 10步
   - 评估推理速度和性能权衡

### 零样本泛化测试

在训练时使用特定颜色（如红、蓝、绿），测试时使用未见过的颜色（如紫、橙）：

```bash
python scripts/eval.py \
    --checkpoint checkpoints/pick_cube_model/best.pth \
    --instruction "Pick up the purple cube"  # 未见过的颜色
```

## 常见问题

### Q: CLIP 模型下载失败？

A: 设置代理或手动下载：
```bash
# 手动下载到 ~/.cache/clip/
wget https://openaipublic.azureedge.net/clip/models/...
```

### Q: GPU 内存不足？

A: 减小 batch size 或模型规模：
```yaml
training:
  batch_size: 16  # 从 32 减小到 16

model:
  unet_base_channels: 128  # 从 256 减小到 128
```

### Q: 训练不收敛？

A: 检查以下几点：
- 数据质量（专家演示成功率）
- 学习率（尝试 1e-5 到 1e-3）
- 动作归一化（确保 `normalize_actions: true`）

## 项目结构说明

```
LCDP-Sim/
├── lcdp/                      # 核心代码库
│   ├── models/                # 模型定义
│   │   ├── diffusion_policy.py  # 主模型
│   │   ├── vision_encoder.py    # 视觉编码器
│   │   ├── language_encoder.py  # 语言编码器
│   │   ├── unet1d.py            # 1D U-Net
│   │   └── conditioning.py      # FiLM/CrossAttn
│   ├── data/                  # 数据处理
│   │   └── dataset.py
│   └── envs/                  # 环境和专家
│       └── scripted_expert.py
├── scripts/                   # 可执行脚本
│   ├── train.py              # 训练
│   ├── eval.py               # 评估
│   ├── collect_data.py       # 数据采集
│   └── visualize.py          # 可视化
├── configs/                   # 配置文件
│   ├── train_config.yaml
│   └── env_config.yaml
└── notebooks/                 # Jupyter notebooks
    ├── data_exploration.ipynb
    └── model_analysis.ipynb
```

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{lcdp-sim,
  author = {Your Name},
  title = {LCDP-Sim: Language-Conditioned Diffusion Policy for Robot Manipulation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/LCDP-Sim}
}
```

## 参考文献

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et al., RSS 2023)
- [CLIP](https://openai.com/research/clip) (Radford et al., ICML 2021)
- [ManiSkill2](https://github.com/haosulab/ManiSkill2)
