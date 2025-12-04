# LCDP-Sim Contributing Guide

感谢您对 LCDP-Sim 项目的关注！

## 开发设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/LCDP-Sim.git
cd LCDP-Sim

# 安装开发依赖
pip install -e ".[dev]"

# 安装 pre-commit hooks
pre-commit install
```

## 代码规范

### Python 代码风格

- 使用 [Black](https://github.com/psf/black) 格式化代码
- 使用 [isort](https://pycqa.github.io/isort/) 排序导入
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南

```bash
# 格式化代码
black lcdp/
isort lcdp/

# 检查代码质量
flake8 lcdp/
```

### 提交信息规范

使用清晰的提交信息：

```
feat: 添加新的视觉编码器支持
fix: 修复数据加载时的内存泄漏
docs: 更新README中的安装说明
test: 添加扩散模型的单元测试
```

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_models.py

# 带覆盖率报告
pytest tests/ --cov=lcdp --cov-report=html
```

## 添加新功能

1. 创建新分支: `git checkout -b feature/your-feature-name`
2. 实现功能并添加测试
3. 确保所有测试通过
4. 提交 Pull Request

## 报告问题

使用 [GitHub Issues](https://github.com/yourusername/LCDP-Sim/issues) 报告bug或提出功能请求。

请包含：
- 问题的清晰描述
- 重现步骤
- 预期行为 vs. 实际行为
- 系统信息（OS、Python版本、CUDA版本等）

## 联系方式

- Email: your.email@example.com
- GitHub: @yourusername
