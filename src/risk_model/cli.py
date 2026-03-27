import argparse
import json
from datetime import datetime
from pathlib import Path

from .trainer import ModelTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-model",
        description="Risk model training CLI",
    )
    parser.add_argument("data_path", help="Input dataset path: .pkl / .csv / .xlsx / .xls")
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "default_config.json"),
        help="Config path, default: configs/default_config.json",
    )
    parser.add_argument(
        "--model-type",
        default="logistic",
        choices=["logistic", "lightgbm", "xgboost"],
        help="Model type to train",
    )
    return parser


def run_cli(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_path = Path(args.data_path)
    config_path = Path(args.config)

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    run_suffix = datetime.now().strftime("%H%M")
    project_root = Path(__file__).resolve().parents[2]
    config["output_dir"] = str(project_root / "output" / "cli")
    config["model_prefix"] = f"model_{run_suffix}"

    print("=" * 60)
    print("min_model training entry")
    print("=" * 60)
    print(f"Data file: {data_path}")
    print(f"Config file: {config_path}")
    print(f"Model type: {args.model_type}")

    trainer = ModelTrainer(config=config)
    data = trainer.load_data(data_path)
    _, metrics, var_list = trainer.train_scorecard(data=data, model_type=args.model_type)

    print()
    print("Training finished. Summary:")
    print(f"  Test AUC : {metrics.get('test_auc', 0):.4f}")
    print(f"  Test KS  : {metrics.get('test_ks', 0):.4f}")
    print(f"  OOT AUC  : {metrics.get('oot_auc', 0):.4f}")
    print(f"  OOT KS   : {metrics.get('oot_ks', 0):.4f}")
    print(f"  Var count: {len(var_list)}")
    print(f"  Output   : {config.get('output_dir', './output')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
