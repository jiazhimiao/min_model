#!/usr/bin/env python3
"""Analyze all score and feature columns, then enrich with a feature dictionary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Analyze all features and join their business meanings.")
    parser.add_argument("data_path", help="Input dataset path: .pkl / .csv / .xlsx / .xls")
    parser.add_argument("--dictionary-path", required=True, help="Feature dictionary workbook path")
    parser.add_argument("--status-column", default="settleStatus", help="Status column used to derive overdue targets")
    parser.add_argument("--target-mode", choices=["first_overdue", "current_overdue"], default="first_overdue")
    parser.add_argument("--min-sample", type=int, default=100, help="Minimum valid sample per variable")
    parser.add_argument("--sample-rows", type=int, default=0, help="Optional row cap for faster trial runs")
    parser.add_argument("--binning-scope", choices=["none", "scores", "all"], default="scores", help="Depth of binning after the full inventory pass")
    parser.add_argument("--max-binning-features", type=int, default=300, help="Maximum number of non-score features to bin when binning-scope=all")
    parser.add_argument("--output-dir", default="output/feature_inventory_analysis", help="Directory for outputs")
    return parser


def load_table(path: Path, sample_rows: int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        frame = pd.read_pickle(path)
        return frame.head(sample_rows) if sample_rows > 0 else frame
    if suffix == ".csv":
        if sample_rows > 0:
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=sample_rows)
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
        return frame.head(sample_rows) if sample_rows > 0 else frame
    raise ValueError(f"Unsupported file type: {suffix}")


def load_dictionary(path: Path) -> pd.DataFrame:
    dictionary = pd.read_excel(path)
    feature_col = dictionary.columns[0]
    meaning_col = dictionary.columns[1]
    result = dictionary[[feature_col, meaning_col]].copy()
    result.columns = ["feature", "meaning"]
    result["feature"] = result["feature"].astype(str)
    return result.dropna(subset=["feature"]).drop_duplicates(subset=["feature"], keep="first")


def normalize_status(frame: pd.DataFrame, status_column: str) -> pd.DataFrame:
    normalized = frame.copy()
    numeric_status = pd.to_numeric(normalized[status_column], errors="coerce")
    normalized[status_column] = numeric_status.replace(SETTLESTATUS_NORMALIZATION)
    normalized[f"{status_column}_normalized"] = normalized[status_column]
    normalized["target_first_overdue"] = normalized[status_column].isin([0, 3]).astype("Int64")
    normalized["target_current_overdue"] = normalized[status_column].isin([0]).astype("Int64")
    return normalized


def is_monotonic(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    if len(values) <= 2:
        return True
    inc = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    dec = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return inc or dec


def classify_variable(name: str) -> str:
    if name.startswith("MA") and not name.startswith("MAIN"):
        return "score"
    return "feature"


def select_feature_columns(frame: pd.DataFrame, status_column: str, target_column: str) -> list[str]:
    excluded = {
        status_column,
        f"{status_column}_normalized",
        "target_first_overdue",
        "target_current_overdue",
        "settleTime",
        "settledCapital",
        "settledCapitalInterest",
        target_column,
    }
    return [col for col in frame.columns if col not in excluded]


def profile_variable(frame: pd.DataFrame, variable: str, dictionary_map: dict[str, str]) -> dict:
    series = frame[variable]
    valid = series.dropna()
    numeric = pd.to_numeric(series, errors="coerce")
    unique_values = valid.astype(str).drop_duplicates().head(3).tolist()
    return {
        "feature": variable,
        "feature_type": classify_variable(variable),
        "meaning": dictionary_map.get(variable, "未在特征码汇总中找到"),
        "dtype": str(series.dtype),
        "valid_rows": int(valid.shape[0]),
        "missing_rows": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()),
        "distinct_count": int(valid.nunique(dropna=True)),
        "numeric_nonnull": int(numeric.notna().sum()),
        "numeric_ratio": float(numeric.notna().mean()),
        "sample_values": " | ".join(unique_values),
        "constant_flag": bool(valid.nunique(dropna=True) <= 1),
        "bin_count": 0,
        "iv": 0.0,
        "monotonic": False,
        "max_bin_share": 0.0,
        "top_bad_rate": None,
        "bottom_bad_rate": None,
        "recommendation": "profile_only",
    }


def should_bin(profile: dict, binning_scope: str, binned_feature_count: int, max_binning_features: int) -> bool:
    if profile["valid_rows"] < 100 or profile["constant_flag"]:
        return False
    if binning_scope == "none":
        return False
    if binning_scope == "scores":
        return profile["feature_type"] == "score"
    if profile["feature_type"] == "score":
        return True
    if max_binning_features > 0 and binned_feature_count >= max_binning_features:
        return False
    return True


def analyze_variable(frame: pd.DataFrame, variable: str, target_column: str, min_sample: int, binning: WOEBinning, profile: dict) -> tuple[dict, pd.DataFrame | None]:
    valid = frame[[variable, target_column]].dropna()
    result_base = profile.copy()
    valid_rows = int(len(valid))
    if valid_rows < min_sample:
        result_base["recommendation"] = "insufficient_sample"
        return result_base, None

    try:
        binned = binning.fit(frame, variable, target_column)
    except Exception as exc:
        result_base["recommendation"] = f"binning_error:{type(exc).__name__}"
        return result_base, None
    if binned is None or binned.empty:
        result_base["recommendation"] = "binning_failed"
        return result_base, None

    numeric_bins = binned.loc[binned["bin_range"] != "missing"].copy()
    monotonic = is_monotonic(numeric_bins["bad_rate"])
    iv_value = float(binned["iv"].sum())
    max_bin_share = float(binned["pct"].max())
    recommendation = "review"
    if monotonic and iv_value >= 0.05 and max_bin_share < 0.7:
        recommendation = "priority_review"
    elif monotonic and iv_value >= 0.02:
        recommendation = "secondary_review"
    elif classify_variable(variable) == "score":
        recommendation = "score_monitoring"
    else:
        recommendation = "low_priority"

    result_base.update(
        {
            "bin_count": int(len(binned)),
            "iv": iv_value,
            "monotonic": monotonic,
            "max_bin_share": max_bin_share,
            "top_bad_rate": float(numeric_bins["bad_rate"].max()) if not numeric_bins.empty else None,
            "bottom_bad_rate": float(numeric_bins["bad_rate"].min()) if not numeric_bins.empty else None,
            "recommendation": recommendation,
        }
    )
    detail = binned.copy()
    detail.insert(0, "feature", variable)
    detail.insert(1, "feature_type", classify_variable(variable))
    return result_base, detail


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    dictionary_path = Path(args.dictionary_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = normalize_status(load_table(data_path, args.sample_rows), args.status_column)
    dictionary = load_dictionary(dictionary_path)
    dictionary_map = dict(zip(dictionary["feature"], dictionary["meaning"]))
    target_column = "target_first_overdue" if args.target_mode == "first_overdue" else "target_current_overdue"

    variables = select_feature_columns(data, args.status_column, target_column)
    binning = WOEBinning()
    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    binned_feature_count = 0

    for variable in variables:
        profile = profile_variable(data, variable, dictionary_map)
        if should_bin(profile, args.binning_scope, binned_feature_count, args.max_binning_features):
            summary, detail = analyze_variable(
                frame=data,
                variable=variable,
                target_column=target_column,
                min_sample=args.min_sample,
                binning=binning,
                profile=profile,
            )
            if detail is not None and profile["feature_type"] == "feature":
                binned_feature_count += 1
            if detail is not None:
                detail_frames.append(detail)
        else:
            summary = profile
            if summary["valid_rows"] < args.min_sample:
                summary["recommendation"] = "insufficient_sample"
            elif summary["constant_flag"]:
                summary["recommendation"] = "constant_or_single_value"
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["feature_type", "recommendation", "monotonic", "iv", "valid_rows"],
        ascending=[True, True, False, False, False],
    )

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    if not detail_df.empty:
        detail_df = detail_df.merge(dictionary, how="left", on="feature")
        detail_df["meaning"] = detail_df["meaning"].fillna("未在特征码汇总中找到")

    score_summary = summary_df.loc[summary_df["feature_type"] == "score"].copy()
    feature_summary = summary_df.loc[summary_df["feature_type"] == "feature"].copy()

    summary_path = output_dir / "feature_inventory_summary.csv"
    score_path = output_dir / "score_candidates.csv"
    feature_path = output_dir / "feature_candidates.csv"
    detail_path = output_dir / "feature_inventory_bin_details.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    score_summary.to_csv(score_path, index=False, encoding="utf-8-sig")
    feature_summary.to_csv(feature_path, index=False, encoding="utf-8-sig")
    if not detail_df.empty:
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"Analyzed variables: {len(summary_df)}")
    print(f"Sample rows used: {len(data)}")
    print(f"Score candidates: {len(score_summary)}")
    print(f"Feature candidates: {len(feature_summary)}")
    print(f"Summary output: {summary_path}")
    print(f"Score output: {score_path}")
    print(f"Feature output: {feature_path}")
    if not detail_df.empty:
        print(f"Bin detail output: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
