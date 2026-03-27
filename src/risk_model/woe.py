# -*- coding: utf-8 -*-
"""WOE binning wrapper built from the existing tool-library implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils.woe_tools import calc_woe_details, transform as transform_woe, var_bin


def _parse_bin_range(bin_range: str) -> tuple[float, float]:
    if not isinstance(bin_range, str) or bin_range == "missing":
        return (np.nan, np.nan)

    values = re.sub(r"[\(\)\[\]]", "", bin_range).split(",")
    left = values[0].strip()
    right = values[1].strip()
    ll = -np.inf if left == "-inf" else float(left)
    ul = np.inf if right == "inf" else float(right)
    return ll, ul


@dataclass
class WOEBinning:
    """WOE helper that delegates binning and WOE calculation to tool functions."""

    min_bins: int = 2
    max_bins: int = 7
    bin_pct: float = 0.05
    raw_bin_multiplier: int = 3
    prefer_monotonic: bool = True
    bin_info: dict = field(default_factory=dict)
    woe_frames: dict = field(default_factory=dict)

    def _is_monotonic(self, details_df: pd.DataFrame) -> bool:
        if details_df is None or details_df.empty:
            return False
        numeric_df = details_df.loc[details_df["bin_range"] != "missing"].copy()
        if len(numeric_df) <= 2:
            return True
        bad_rate = numeric_df["bad_rate"].astype(float).tolist()
        inc = all(bad_rate[i] <= bad_rate[i + 1] for i in range(len(bad_rate) - 1))
        dec = all(bad_rate[i] >= bad_rate[i + 1] for i in range(len(bad_rate) - 1))
        return inc or dec

    def _build_numeric_candidate(self, valid_df, var, y_name, max_rawbin, diff):
        bindata = var_bin(
            indata=valid_df,
            testing=None,
            var=var,
            yname=y_name,
            bintype="IV",
            bin_param={
                "type_rawbin": "freq",
                "max_rawbin": max_rawbin,
                "bin_pct": self.bin_pct,
                "diff": diff,
            },
            outlier=None,
            char_flag=False,
        )
        if bindata is None or bindata.empty:
            return None
        details_df, transform_df = calc_woe_details(
            indata=valid_df,
            yname=y_name,
            bindata=bindata,
            var_value=var,
        )
        if details_df is None or details_df.empty or len(details_df) < self.min_bins:
            return None
        if len(details_df) > self.max_bins:
            return None
        return details_df, transform_df

    def fit(self, data, var, y_name):
        df = data[[var, y_name]].copy()
        valid_df = df.dropna(subset=[var, y_name]).reset_index(drop=True)
        if len(valid_df) < 100:
            return None

        char_flag = (
            valid_df[var].dtype == "object"
            or valid_df[var].dtype.name == "category"
            or valid_df[var].nunique() <= 5
        )
        if char_flag:
            bindata = pd.DataFrame({"LL": sorted(valid_df[var].dropna().unique().tolist())})
            bindata["VAR"] = var
            bindata["IDVAR"] = f"IV_var({var})grpmethod(freq)"
            details_df, transform_df = calc_woe_details(
                indata=valid_df,
                yname=y_name,
                bindata=bindata,
                var_value=var,
            )
        else:
            candidates = []
            raw_bin_candidates = sorted(
                {
                    max(self.max_bins * self.raw_bin_multiplier, 10),
                    max(self.max_bins * max(self.raw_bin_multiplier - 1, 2), 8),
                    max(self.max_bins + 4, 8),
                },
                reverse=True,
            )
            for max_rawbin in raw_bin_candidates:
                for diff in [0, 0.02, 0.05]:
                    candidate = self._build_numeric_candidate(valid_df, var, y_name, max_rawbin=max_rawbin, diff=diff)
                    if candidate is None:
                        continue
                    cand_details_df, cand_transform_df = candidate
                    candidate_iv = float(cand_details_df["iv"].sum())
                    candidate_monotonic = self._is_monotonic(cand_details_df)
                    candidates.append(
                        {
                            "details_df": cand_details_df,
                            "transform_df": cand_transform_df,
                            "iv": candidate_iv,
                            "is_monotonic": candidate_monotonic,
                            "bin_count": len(cand_details_df),
                        }
                    )

            if not candidates:
                return None
            candidates.sort(
                key=lambda item: (
                    1 if (item["is_monotonic"] and self.prefer_monotonic) else 0,
                    item["iv"],
                    -item["bin_count"],
                ),
                reverse=True,
            )
            best_candidate = candidates[0]
            details_df = best_candidate["details_df"]
            transform_df = best_candidate["transform_df"]

        if details_df is None or details_df.empty or len(details_df) < self.min_bins:
            return None

        ll_ul = details_df["bin_range"].apply(_parse_bin_range)
        details_df["LL"] = ll_ul.apply(lambda x: x[0])
        details_df["UL"] = ll_ul.apply(lambda x: x[1])
        details_df["bin_id"] = range(len(details_df))

        final_df = details_df[
            [
                "VAR",
                "bin_id",
                "LL",
                "UL",
                "count",
                "count_good",
                "count_bad",
                "pct",
                "bad_rate",
                "woe",
                "iv",
                "_labels_",
                "bin_range",
            ]
        ].copy()

        self.bin_info[var] = final_df
        self.woe_frames[var] = transform_df.copy()
        return final_df

    def transform(self, data, var):
        if var not in self.woe_frames:
            return data[var]

        transform_input = data[[var]].copy()
        transform_output = transform_woe(
            indata=transform_input,
            woedata=self.woe_frames[var],
            var="VAR",
            bin_range="bin_range",
            woevar="woe",
            suffix="_woe",
        )
        return transform_output[f"{var}_woe"]

    def get_iv(self, var):
        if var not in self.bin_info:
            return 0
        return self.bin_info[var]["iv"].sum()
