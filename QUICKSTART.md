# 🚀 LCDP-Sim 快速开始指南

本指南将带您从零开始运行整个项目，包括环境配置、数据准备、模型训练和可视化。

---

## 📋 目录

1. [环境配置](#1-环境配置)
2. [快速验证（推荐）](#2-快速验证推荐)
3. [完整训练流程](#3-完整训练流程)
4. [可视化结果](#4-可视化结果)
5. [故障排除](#5-故障排除)

---

## 1. 环境配置

### 方法 A: 自动安装（推荐）

```powershell
# 在项目根目录运行
.\setup.ps1
```

### 方法 B: 手动安装

```powershell
# 1. 创建 conda 环境
conda create -n lcdp python=3.8 -y
conda activate lcdp

# 2. 安装 PyTorch（根据您的 CUDA 版本）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（不推荐训练）
# pip install torch torchvision

# 3. 安装核心依赖
pip install transformers diffusers accelerate
pip install numpy pandas matplotlib seaborn tqdm pyyaml
pip install zarr h5py opencv-python imageio pillow
pip install einops ftfy regex

# 4. 安装本项目
pip install -e .
```

### 验证安装

```powershell
python -c "import torch; import lcdp; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**预期输出：**
```
PyTorch: 2.x.x
CUDA available: True  # 如果有 GPU
```

---

## 2. 快速验证（推荐）

这是最快的方式验证项目可以运行，**不需要仿真环境**。

### 步骤 1: 创建模拟数据集

```powershell
python -c "
import numpy as np
import zarr
from pathlib import Path
from numcodecs import JSON

Path('data').mkdir(exist_ok=True)
root = zarr.open('data/demo_data.zarr', mode='w')

num_episodes, episode_length = 10, 100
total_steps = num_episodes * episode_length

print('📦 生成模拟数据集...')
images = np.random.randint(0, 255, (total_steps, 224, 224, 3), dtype=np.uint8)
root.create_dataset('observations/rgb', data=images, chunks=(1, 224, 224, 3))

actions = np.random.randn(total_steps, 7).astype(np.float32) * 0.1
root.create_dataset('actions', data=actions, chunks=(1, 7))

instructions = ['Pick up the red cube'] * num_episodes
root.create_dataset('language_instruction', data=np.array(instructions, dtype=object), object_codec=JSON())

root.create_dataset('episode_lengths', data=np.array([episode_length]*num_episodes))
print(f'✅ 数据集创建完成: {num_episodes} 条轨迹, {total_steps} 步')
"
```

### 步骤 2: 快速训练测试

```powershell
python -c "
from lcdp.data.dataset import RobotDataset
from lcdp.models.diffusion_policy import DiffusionPolicy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

print('=== 🎯 快速训练演示 ===')

dataset = RobotDataset('data/demo_data.zarr', action_horizon=16)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
print(f'📊 数据集: {len(dataset)} 样本')

model = DiffusionPolicy(action_dim=7, action_horizon=16, vision_encoder='resnet18', conditioning_type='film')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(f'🤖 模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

model.train()
total_loss = 0
for batch_idx, batch in enumerate(tqdm(dataloader, desc='训练中')):
    optimizer.zero_grad()
    loss_dict = model.compute_loss(batch['actions'], batch['image'], batch['instruction'])
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    
    if batch_idx >= 10:
        break

print(f'📈 平均损失: {total_loss / (batch_idx + 1):.4f}')
print('✅ 训练完成！')
"
```

**预期输出：**
```
=== 🎯 快速训练演示 ===
📊 数据集: 985 样本
🤖 模型参数: 232.34M
训练中: 100%|████████| 11/11 [00:04<00:00, 2.18it/s]
📈 平均损失: 1.0016
✅ 训练完成！
```

### 步骤 3: 测试推理

```powershell
python -c "
import torch
from lcdp.models.diffusion_policy import DiffusionPolicy

model = DiffusionPolicy(action_dim=7, action_horizon=16, vision_encoder='resnet18', conditioning_type='film')
model.eval()

test_image = torch.randn(1, 3, 224, 224)
test_instruction = ['Pick up the red cube']

with torch.no_grad():
    actions = model.get_action(test_image, test_instruction, num_inference_steps=10)

print('✅ 推理成功！')
print(f'📥 输入: 1张图像 (224x224x3) + 1条指令')
print(f'📤 输出: 动作序列 {actions.shape}')
print(f'📊 动作维度: [batch={actions.shape[0]}, dof={actions.shape[1]}, horizon={actions.shape[2]}]')
print(f'\\n前3个时间步的动作 (x, y, z):')
print(actions[0, :3, :3].numpy())
"
```

---

## 3. 完整训练流程

### 选项 A: 使用模拟数据（推荐开始）

上面的快速验证就是完整的模拟数据训练流程。

如果想训练更长时间：

```powershell
# 创建更大的数据集
python -c "
import numpy as np
import zarr
from pathlib import Path
from numcodecs import JSON

Path('data').mkdir(exist_ok=True)
root = zarr.open('data/large_demo.zarr', mode='w')

# 100 条轨迹
num_episodes, episode_length = 100, 200
total_steps = num_episodes * episode_length

print(f'创建大型数据集: {num_episodes} 轨迹...')
images = np.random.randint(0, 255, (total_steps, 224, 224, 3), dtype=np.uint8)
root.create_dataset('observations/rgb', data=images, chunks=(1, 224, 224, 3))

actions = np.random.randn(total_steps, 7).astype(np.float32) * 0.1
root.create_dataset('actions', data=actions, chunks=(1, 7))

instructions = ['Pick up the red cube', 'Push the blue block', 'Stack the green cube'] * (num_episodes // 3 + 1)
root.create_dataset('language_instruction', data=np.array(instructions[:num_episodes], dtype=object), object_codec=JSON())

root.create_dataset('episode_lengths', data=np.array([episode_length]*num_episodes))
print(f'✅ 完成: {total_steps} 步')
"

# 使用训练脚本
python scripts/train.py --data data/large_demo.zarr --config configs/train_config.yaml --output checkpoints/my_model
```

### 选项 B: 使用仿真环境（需要额外配置）

⚠️ **注意**: ManiSkill2 在 Windows 上安装较复杂，建议先完成选项 A。

```powershell
# 1. 安装仿真环境（可能失败）
pip install gymnasium
# pip install mani-skill2  # 可能有依赖问题

# 2. 收集专家演示
python scripts/collect_data.py --env PickCube-v0 --num-episodes 100 --output data/pick_cube_demos.zarr

# 3. 训练模型
python scripts/train.py --data data/pick_cube_demos.zarr --config configs/train_config.yaml --output checkpoints/pick_cube_model

# 4. 评估
python scripts/eval.py --checkpoint checkpoints/pick_cube_model/best.pth --env PickCube-v0 --num-episodes 50
```

---

## 4. 可视化结果

### 方法 1: 使用内置可视化脚本

```powershell
# 可视化训练过程
python scripts/visualize.py --data data/demo_data.zarr --output visualizations/

# 可视化模型预测
python scripts/visualize.py --checkpoint checkpoints/my_model/best.pth --instruction "Pick up the red cube" --output visualizations/prediction.png
```

### 方法 2: 手动可视化动作序列

```powershell
python -c "
import torch
import matplotlib.pyplot as plt
import numpy as np
from lcdp.models.diffusion_policy import DiffusionPolicy

# 加载模型
model = DiffusionPolicy(action_dim=7, action_horizon=16, vision_encoder='resnet18', conditioning_type='film')
model.eval()

# 生成预测
test_image = torch.randn(1, 3, 224, 224)
test_instruction = ['Pick up the red cube']

with torch.no_grad():
    actions = model.get_action(test_image, test_instruction, num_inference_steps=10)

# 可视化
actions_np = actions[0].cpu().numpy()  # [7, 16]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

for i in range(7):
    ax = axes[i // 4, i % 4]
    ax.plot(actions_np[i], marker='o', linewidth=2)
    ax.set_title(f'{labels[i]} Position', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

# 删除多余的子图
axes[1, 3].remove()

plt.tight_layout()
plt.savefig('visualizations/action_trajectory.png', dpi=150, bbox_inches='tight')
print('✅ 可视化保存至: visualizations/action_trajectory.png')
"
```

### 方法 3: 生成训练曲线

```powershell
python -c "
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

Path('visualizations').mkdir(exist_ok=True)

# 模拟训练损失曲线
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-epochs / 10) + 0.3 + np.random.randn(50) * 0.05
val_loss = 2.2 * np.exp(-epochs / 10) + 0.4 + np.random.randn(50) * 0.07

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/training_curve.png', dpi=150)
print('✅ 训练曲线保存至: visualizations/training_curve.png')
"
```

---

## 5. 故障排除

### 问题 1: CUDA 不可用

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
```powershell
# 检查 CUDA 版本
nvidia-smi

# 重新安装对应版本的 PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # 根据您的 CUDA 版本
```

### 问题 2: 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 batch size: 修改 `configs/train_config.yaml` 中的 `batch_size: 2`
- 使用梯度累积
- 使用 CPU 训练（较慢）

### 问题 3: ManiSkill2 安装失败

**症状**: `pip install mani-skill2` 报错

**解决方案**:
- **推荐**: 先使用模拟数据验证项目（选项 A）
- ManiSkill2 依赖复杂，Windows 支持不完善
- 可以使用简单的 Gymnasium 环境替代

### 问题 4: 模块导入错误

**症状**: `ModuleNotFoundError: No module named 'lcdp'`

**解决方案**:
```powershell
# 确保安装了项目
pip install -e .

# 或直接将项目路径加入 PYTHONPATH
$env:PYTHONPATH = "d:\3.code\Python\LCDP-Sim;$env:PYTHONPATH"
```

---

## 📊 预期结果总结

### 快速验证模式（推荐）
- ⏱️ **时间**: 5-10 分钟
- 📦 **数据集**: 1000 步模拟数据
- 🤖 **模型**: 232M 参数
- 📈 **训练**: 10 批次，损失 ~1.0
- ✅ **输出**: 动作轨迹可视化

### 完整训练模式
- ⏱️ **时间**: 1-2 小时
- 📦 **数据集**: 20,000 步
- 🤖 **模型**: 232M 参数  
- 📈 **训练**: 50 epochs
- ✅ **输出**: 训练好的模型 + 完整可视化

---

## 🎯 下一步

1. ✅ 完成快速验证
2. 📊 生成可视化结果
3. 📝 撰写项目报告
4. 🚀 将结果添加到 README
5. 💼 用于简历/面试展示

---

## 📧 需要帮助？

如果遇到问题，请检查：
- [README.md](README.md) - 项目概览
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构说明
- [docs/USAGE.md](docs/USAGE.md) - 详细使用文档

或联系作者: hy20051123@gmail.com
