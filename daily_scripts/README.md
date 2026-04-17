# 智能特征分箱分析平台（daily_scripts）

本目录包含两套入口：

- 命令行脚本：`feature_woe_analysis.py`
- Web 页面（Streamlit）：`feature_woe_web_app.py`

用于对特征做 WOE 分箱分析、分组汇总、Excel 输出与历史管理。

## 1. 主要文件

- `feature_woe_analysis.py`：核心分析脚本（读取数据、分层、分箱、汇总、写 Excel）。
- `feature_woe_web_app.py`：Web 交互页面（上传文件、参数选择、进度展示、结果下载）。
- `config/feature_sheet_output_config.json`：Sheet 输出开关配置（逐项 true/false）。
- `config/feature_display_config.json`：展示配置（如是否显示中文解释）。
- `config/模型分特征码汇总.xlsx`：特征码中文解释映射（本地数据，默认不建议入库）。

## 2. 命令行运行

```bash
cd d:\Trae_pro\min_model\daily_scripts
python feature_woe_analysis.py <输入文件> [status_source]
```

示例：

```bash
python feature_woe_analysis.py "d:\data\sample.xlsx" 1
```

等价参数写法：

```bash
python feature_woe_analysis.py "d:\data\sample.xlsx" --status-source 1
```

`status_source` 含义：

- `0 -> settleStatus`
- `1 -> settleStatus_1`
- `2 -> settleStatus_2`
- `3 -> settleStatus_3`

## 3. Web 运行

```bash
cd d:\Trae_pro\min_model\daily_scripts
python -m streamlit run feature_woe_web_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

打开：`http://127.0.0.1:8501`

## 4. 输出目录说明

- `feature_analysis_output/web_runtime/`：运行时中间目录（可清理）。
- `feature_analysis_output/web_runs/`：历史任务归档目录（按需清理）。
- `feature_analysis_output/<数据集名>/`：命令行输出目录（按需保留）。

## 5. 建议上传到 Git 的内容

建议仅上传：

- `feature_woe_analysis.py`
- `feature_woe_web_app.py`
- `README.md`
- `.gitignore`
- `tools/`（若这些函数依赖已调整）
- `config/*.json`（不含敏感配置时）

不建议上传：

- 大体量原始数据（`*.xlsx/*.csv/*.parquet`）
- `feature_analysis_output/` 全部运行产物
- `runtime/`、`__pycache__/`
- `config/模型分特征码汇总.xlsx`（若含业务敏感信息）

## 6. 清理建议

优先清理中间态目录：

- `feature_analysis_output/web_runtime/`
- `feature_analysis_output/web_runs/`（若历史不需要）
- `runtime/`
- `__pycache__/`
- `feature_woe_web_r6cplldw/`（异常临时目录，若不再占用建议清理）

> 如果某目录清理失败，通常是文件句柄占用。先停止 Streamlit / Python 进程后再删。
