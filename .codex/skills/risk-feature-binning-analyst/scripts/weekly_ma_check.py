#!/usr/bin/env python3
"""Weekly review for MA-prefixed scores."""

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
    parser = argparse.ArgumentParser(description="Run weekly MA score ordering checks.")
    parser.add_argument("data_path", help="Input dataset path: .pkl / .csv / .xlsx / .xls")
    parser.add_argument("--target", required=True, help="Binary target column name")
    parser.add_argument("--prefix", default="MA", help="Score prefix to inspect")
    parser.add_argument("--scores", nargs="*", help="Explicit score columns to inspect")
    parser.add_argument("--output-dir", default="output/ma_weekly_check", help="Directory for csv and markdown outputs")
    parser.add_argument("--min-sample", type=int, default=100, help="Minimum valid rows required per score")
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


def separation_gap(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    if len(values) <= 1:
        return 0.0
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    return float(min(diffs))


def classify_score(monotonic: bool, min_gap: float, max_share: float) -> str:
    if monotonic and min_gap >= 0.01 and max_share < 0.7:
        return "stable_ordering_and_separation"
    if monotonic:
        return "ordering_stable_separation_weakened"
    return "ordering_broken_investigate_drift"


def build_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# MA Score Weekly Review",
        "",
        "| score | valid_rows | bins | iv | monotonic | min_gap | max_bin_share | conclusion |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.score} | {row.valid_rows} | {row.bin_count} | {row.iv:.4f} | "
            f"{'yes' if row.monotonic else 'no'} | {row.min_adjacent_gap:.4f} | "
            f"{row.max_bin_share:.4f} | {row.conclusion} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_settlestatus(load_table(data_path), args.status_column)
    if args.target not in df.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    if args.scores:
        scores = [col for col in args.scores if col in df.columns]
    else:
        scores = [col for col in df.columns if col.startswith(args.prefix)]
    if not scores:
        raise SystemExit("No MA scores found for review.")

    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    binning = WOEBinning()

    for score in scores:
        valid = df[[score, args.target]].dropna()
        if len(valid) < args.min_sample:
            summary_rows.append(
                {
                    "score": score,
                    "valid_rows": len(valid),
                    "bin_count": 0,
                    "iv": 0.0,
                    "monotonic": False,
                    "min_adjacent_gap": 0.0,
                    "max_bin_share": 0.0,
                    "conclusion": "insufficient_sample",
                }
            )
            continue

        result = binning.fit(df, score, args.target)
        if result is None or result.empty:
            summary_rows.append(
                {
                    "score": score,
                    "valid_rows": len(valid),
                    "bin_count": 0,
                    "iv": 0.0,
                    "monotonic": False,
                    "min_adjacent_gap": 0.0,
                    "max_bin_share": 0.0,
                    "conclusion": "binning_failed",
                }
            )
            continue

        numeric_bins = result.loc[result["bin_range"] != "missing"].copy()
        monotonic = is_monotonic(numeric_bins["bad_rate"])
        min_gap = separation_gap(numeric_bins["bad_rate"])
        max_bin_share = float(result["pct"].max())
        conclusion = classify_score(monotonic, min_gap, max_bin_share)

        summary_rows.append(
            {
                "score": score,
                "valid_rows": len(valid),
                "bin_count": len(result),
                "iv": float(result["iv"].sum()),
                "monotonic": monotonic,
                "min_adjacent_gap": min_gap,
                "max_bin_share": max_bin_share,
                "conclusion": conclusion,
            }
        )

        detail_frame = result.copy()
        detail_frame.insert(0, "score", score)
        detail_frames.append(detail_frame)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["monotonic", "iv", "valid_rows"],
        ascending=[False, False, False],
    )
    summary_path = output_dir / "ma_score_summary.csv"
    detail_path = output_dir / "ma_score_bin_details.csv"
    report_path = output_dir / "ma_weekly_review.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_markdown(summary_df), encoding="utf-8")

    print(f"Reviewed {len(summary_df)} MA scores")
    print(f"Summary output: {summary_path}")
    if detail_frames:
        print(f"Bin details output: {detail_path}")
    print(f"Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
