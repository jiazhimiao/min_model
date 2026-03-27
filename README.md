# min_model

`min_model` 是从原始仓库中提炼出来的最小可运行建模项目。
它不会改动原有目录结构，而是在 [min_model](D:\Trae_pro\model_\min_model) 中单独整理出一套更适合维护、测试和继续演进的主干代码。

## 项目目标

- 不影响原始仓库内容
- 提炼统一训练主干
- 将历史项目、日志、模型产物和大体量数据与主干代码分离
- 为后续模块拆分、补测试和部署保留清晰结构

## 目录结构

```text
min_model/
  src/
    risk_model/
      __init__.py
      trainer.py
      woe.py
      exporters/
      utils/
  tests/
    test_model_trainer.py
  configs/
    default_config.json
  docs/
    cleanup_scope.md
    project_layout.md
    test_result.md
  scripts/
    run_training.py
  legacy/
  .gitignore
  pyproject.toml
  requirements.txt
  README.md
```

## 核心文件说明

- [trainer.py](D:\Trae_pro\model_\min_model\src\risk_model\trainer.py)
  统一训练入口，支持 `logistic`、`lightgbm`、`xgboost`、`ensemble`
- [woe.py](D:\Trae_pro\model_\min_model\src\risk_model\woe.py)
  基于工具函数封装后的 `WOEBinning`
- [woe_tools.py](D:\Trae_pro\model_\min_model\src\risk_model\utils\woe_tools.py)
  从原工具链迁移过来的分箱、WOE、IV 和映射逻辑
- [test_model_trainer.py](D:\Trae_pro\model_\min_model\tests\test_model_trainer.py)
  集成测试脚本
- [default_config.json](D:\Trae_pro\model_\min_model\configs\default_config.json)
  默认配置文件

## 安装方式

在 [min_model](D:\Trae_pro\model_\min_model) 目录下执行：

```bash
pip install -e .
```

如果不需要可编辑安装，也可以执行：

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 准备数据

当前测试默认使用：

- [data.pkl](D:\Trae_pro\model_\min_model\data.pkl)

### 2. 运行集成测试

```bash
python tests/test_model_trainer.py
```

这个脚本会依次训练：

- 逻辑回归评分卡
- LightGBM
- XGBoost

如果需要正式交付口径，当前更推荐使用融合模型 `ensemble`。

### 3. 单独运行训练脚本

```bash
python scripts/run_training.py data.pkl
```

如果你有自己的数据文件，也可以替换成：

```bash
python scripts/run_training.py 你的数据路径.pkl
```

也可以直接使用中文 CLI：

```bash
python -m risk_model.cli data.pkl --model-type logistic --config configs/default_config.json
```

当前推荐的正式交付方式：

```bash
python -m risk_model.cli data.pkl --model-type ensemble --config configs/default_config.json
```

如果已经执行过 `pip install -e .`，还可以直接使用：

```bash
risk-model data.pkl --model-type lightgbm
```

### 4. 在代码中调用

```python
from risk_model import ModelTrainer
import pandas as pd

data = pd.read_pickle("data.pkl")
trainer = ModelTrainer()
model, metrics, var_list = trainer.train_scorecard(
    data=data,
    model_type="logistic",
)
```

## 输出结果

训练完成后，默认会在 [output](D:\Trae_pro\model_\min_model\output) 下生成：

- 模型文件 `.pkl`
- 变量清单 `.csv`
- ROC 图 `.png`
- 训练报告 `.xlsx`
- 评分卡 `.xlsx`
- PMML 文件 `.pmml`（树模型且环境支持时）

## 当前验证情况

已经使用 [data.pkl](D:\Trae_pro\model_\min_model\data.pkl) 跑通过完整测试，详细结果见：

- [test_result.md](D:\Trae_pro\model_\min_model\docs\test_result.md)

## 当前实现说明

- `trainer.py` 中原本内嵌的旧版 `WOEBinning` 已移除
- 当前统一使用 [woe.py](D:\Trae_pro\model_\min_model\src\risk_model\woe.py) 中的工具版实现
- 原仓库中的历史项目、日志、缓存、副本和大文件没有迁入主干代码目录
- 已新增中文 CLI 包装层，方便在不改动主训练逻辑的前提下提供更清晰的命令行使用体验

## 后续建议

- 继续把 `trainer.py` 拆成更细的模块
- 增加真正的单元测试，而不只是集成脚本
- 将配置改成 YAML 或 JSON 驱动
- 为训练、预测、导出分别提供 CLI 子命令
