import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risk_model import ModelTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all_models",
        description="输入一个数据集，批量训练 LR、LightGBM、XGBoost，并输出三套模型和报告。",
    )
    parser.add_argument("data_path", help="输入数据路径，支持 .pkl / .csv / .xlsx / .xls")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default_config.json"),
        help="配置文件路径，默认使用 min_model/configs/default_config.json",
    )
    parser.add_argument(
        "--output-subdir",
        default="batch",
        help="输出子目录，默认写入 min_model/output/batch",
    )
    return parser


def load_data(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(data_path)
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    raise ValueError(f"不支持的数据格式: {data_path}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_path = Path(args.data_path)
    config_path = Path(args.config)
    if not data_path.exists():
        raise SystemExit(f"数据文件不存在: {data_path}")
    if not config_path.exists():
        raise SystemExit(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    print("=" * 60)
    print("min_model 批量训练入口")
    print("=" * 60)
    print(f"数据文件: {data_path}")
    print(f"配置文件: {config_path}")

    data = load_data(data_path)
    print(f"数据加载完成: {len(data)} 条记录, {len(data.columns)} 列")

    run_suffix = datetime.now().strftime("%H%M")
    output_dir = ROOT / "output" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_plan = [
        ("logistic", "lr"),
        ("lightgbm", "lgb"),
        ("xgboost", "xgb"),
    ]
    results = []

    for model_type, short_name in model_plan:
        print("\n" + "=" * 60)
        print(f"开始训练: {model_type}")
        print("=" * 60)
        config = copy.deepcopy(base_config)
        config["output_dir"] = str(output_dir)
        config["model_prefix"] = f"{short_name}_{run_suffix}"

        trainer = ModelTrainer(config=config)
        try:
            _, metrics, var_list = trainer.train_scorecard(data=data.copy(), model_type=model_type)
            results.append(
                {
                    "model_type": model_type,
                    "status": "成功",
                    "test_auc": metrics.get("test_auc", 0),
                    "test_ks": metrics.get("test_ks", 0),
                    "oot_auc": metrics.get("oot_auc", 0),
                    "oot_ks": metrics.get("oot_ks", 0),
                    "var_count": len(var_list),
                    "prefix": config["model_prefix"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "model_type": model_type,
                    "status": f"失败: {e}",
                    "test_auc": None,
                    "test_ks": None,
                    "oot_auc": None,
                    "oot_ks": None,
                    "var_count": None,
                    "prefix": config["model_prefix"],
                }
            )

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / f"summary_{run_suffix}.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="batch_summary", index=False)

    print("\n" + "=" * 60)
    print("批量训练结果汇总")
    print("=" * 60)
    for row in results:
        if row["status"] == "成功":
            print(
                f"{row['model_type']:<10} "
                f"test_auc={row['test_auc']:.4f}  "
                f"test_ks={row['test_ks']:.4f}  "
                f"oot_auc={row['oot_auc']:.4f}  "
                f"oot_ks={row['oot_ks']:.4f}  "
                f"vars={row['var_count']}"
            )
        else:
            print(f"{row['model_type']:<10} {row['status']}")

    print(f"\n汇总文件: {summary_path}")
    print(f"输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
