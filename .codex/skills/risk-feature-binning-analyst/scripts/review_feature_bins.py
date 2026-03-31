#!/usr/bin/env python3
"""Batch review feature binning quality for risk analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_model.woe import WOEBinning  # noqa: E402

SETTLESTATUS_NORMALIZATION = {
    4: 2,
    5: 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review feature bins and monotonicity.")
    parser.add_argument("data_path", help="Input dataset path: .pkl / .csv / .xlsx / .xls")
    parser.add_argument("--target", required=True, help="Binary target column name")
    parser.add_argument("--features", nargs="*", help="Specific feature columns to review")
    parser.add_argument("--prefix", help="Only review columns with this prefix")
    parser.add_argument("--exclude-prefix", nargs="*", default=[], help="Skip columns with these prefixes")
    parser.add_argument("--max-features", type=int, default=0, help="Optional limit on number of reviewed features")
    parser.add_argument("--output-dir", default="output/binning_review", help="Directory for csv outputs")
    parser.add_argument("--min-sample", type=int, default=100, help="Minimum valid rows required per feature")
    parser.add_argument("--status-column", default="settlestatus", help="Repayment status column to normalize when present")
    return parser


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_settlestatus(frame: pd.DataFrame, status_column: str) -> pd.DataFrame:
    if status_column not in frame.columns:
        return frame
    normalized = frame.copy()
    numeric_status = pd.to_numeric(normalized[status_column], errors="coerce")
    normalized[status_column] = numeric_status.replace(SETTLESTATUS_NORMALIZATION)
    normalized[f"{status_column}_normalized"] = normalized[status_column]
    return normalized


def is_monotonic(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    if len(values) <= 2:
        return True
    inc = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    dec = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return inc or dec


def recommend_action(monotonic: bool, iv_value: float, top_bin_share: float, distinct_count: int) -> str:
    if monotonic and iv_value >= 0.05 and top_bin_share < 0.7:
        return "recommend_launch_candidate"
    if monotonic and iv_value >= 0.02:
        return "observe_or_partial_rule"
    if distinct_count <= 20:
        return "manual_review_small_cardinality"
    return "rebin_and_retest"


def select_features(frame: pd.DataFrame, target: str, features: list[str] | None, prefix: str | None, exclude_prefix: list[str]) -> list[str]:
    if features:
        candidates = [col for col in features if col in frame.columns]
    else:
        candidates = [col for col in frame.columns if col != target]
        if prefix:
            candidates = [col for col in candidates if col.startswith(prefix)]
    if exclude_prefix:
        candidates = [col for col in candidates if not any(col.startswith(item) for item in exclude_prefix)]
    return candidates


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_settlestatus(load_table(data_path), args.status_column)
    if args.target not in df.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    features = select_features(df, args.target, args.features, args.prefix, args.exclude_prefix)
    if args.max_features > 0:
        features = features[: args.max_features]
    if not features:
        raise SystemExit("No features selected for review.")

    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    binning = WOEBinning()

    for feature in features:
        valid = df[[feature, args.target]].dropna()
        if len(valid) < args.min_sample:
            summary_rows.append(
                {
                    "feature": feature,
                    "valid_rows": len(valid),
                    "distinct_count": valid[feature].nunique(dropna=True),
                    "bin_count": 0,
                    "iv": 0.0,
                    "monotonic": False,
                    "max_bin_share": 0.0,
                    "recommendation": "insufficient_sample",
                }
            )
            continue

        result = binning.fit(df, feature, args.target)
        if result is None or result.empty:
            summary_rows.append(
                {
                    "feature": feature,
                    "valid_rows": len(valid),
                    "distinct_count": valid[feature].nunique(dropna=True),
                    "bin_count": 0,
                    "iv": 0.0,
                    "monotonic": False,
                    "max_bin_share": 0.0,
                    "recommendation": "binning_failed",
                }
            )
            continue

        numeric_bins = result.loc[result["bin_range"] != "missing"].copy()
        iv_value = float(result["iv"].sum())
        monotonic = is_monotonic(numeric_bins["bad_rate"])
        max_bin_share = float(result["pct"].max())
        distinct_count = int(valid[feature].nunique(dropna=True))
        recommendation = recommend_action(monotonic, iv_value, max_bin_share, distinct_count)

        summary_rows.append(
            {
                "feature": feature,
                "valid_rows": len(valid),
                "distinct_count": distinct_count,
                "bin_count": len(result),
                "iv": iv_value,
                "monotonic": monotonic,
                "max_bin_share": max_bin_share,
                "recommendation": recommendation,
            }
        )

        detail_frame = result.copy()
        detail_frame.insert(0, "feature", feature)
        detail_frames.append(detail_frame)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["monotonic", "iv", "valid_rows"],
        ascending=[False, False, False],
    )
    summary_path = output_dir / "feature_summary.csv"
    detail_path = output_dir / "bin_details.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"Reviewed {len(summary_df)} features")
    print(f"Summary output: {summary_path}")
    if detail_frames:
        print(f"Bin details output: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
