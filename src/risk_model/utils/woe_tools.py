# -*- coding: utf-8 -*-
"""Minimal subset of the existing WOE tool functions migrated into min_model."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy.stats import ks_2samp
from sklearn import model_selection, tree


def get_range(predefined_bins):
    bin1 = ["-inf"] + [str(_) for _ in predefined_bins] + ["inf"]
    bin2 = ["-inf"] + ["{v:.3f}".format(v=_) for _ in predefined_bins] + ["inf"]
    list_v1 = ["(" + bin1[i] + "," + bin1[i + 1] + "]" for i in range(len(predefined_bins) + 1)]
    list_v2 = ["(" + bin2[i] + "," + bin2[i + 1] + "]" for i in range(len(predefined_bins) + 1)]
    return pd.DataFrame({"_labels_": range(len(predefined_bins) + 1), "bin_range": list_v1, "bin_range2": list_v2})


def value_fit(x, predefined_bins):
    df = pd.DataFrame({"X": x, "order": np.arange(x.size)})
    bins = list(predefined_bins)
    if not bins or bins[0] != float("-Inf"):
        bins = np.append(-float("inf"), bins)
    cuts = pd.cut(df["X"], right=True, bins=np.append(bins, float("inf")), labels=np.arange(len(bins)).astype(str))
    df["_labels_"] = np.where(cuts.isnull(), "-9", cuts.astype(str))
    df["_labels_"] = df["_labels_"].astype(int)
    rangeds = get_range(predefined_bins=predefined_bins).drop(["bin_range2"], axis=1)
    return pd.merge(df, rangeds, on="_labels_", how="left")


def _bucket_woe(x):
    t_bad = x["bad"]
    t_good = x["good"]
    t_bad = 0.5 if t_bad == 0 else t_bad
    t_good = 0.5 if t_good == 0 else t_good
    return np.log(t_bad / t_good)


def _calc_woe(df, x="_labels_", y="Y"):
    def math_log1(v):
        return math.log(v) if v != 0 else float("inf")

    stat = df.groupby(x)[y].agg([np.count_nonzero, np.size])
    stat.columns = ["bad", "obs"]
    stat1 = df.groupby(y)[y].agg([np.size]).copy()
    stat1.columns = ["obs"]
    if stat1.shape[0] < 2:
        stat = pd.DataFrame(columns=[x, "bad", "obs", "good", "overdue_rate", "p0", "p1", "woe1", "iv", "t_iv", "woe2", "iv2"])
    else:
        stat["good"] = stat["obs"] - stat["bad"]
        stat["overdue_rate"] = stat["bad"] / stat["obs"]
        stat["p0"] = stat["good"] / stat1.loc[0, "obs"]
        stat["p1"] = stat["bad"] / stat1.loc[1, "obs"]
        if stat.loc[stat.p0 == 0, :].shape[0] == 0:
            stat["woe1"] = stat["p1"] / stat["p0"]
            stat["woe1"] = [math_log1(v) for v in stat["woe1"].values]
            stat["iv"] = stat["woe1"] * (stat["p1"] - stat["p0"])
            stat["t_iv"] = sum(stat["iv"])
        else:
            stat["woe1"] = float("inf")
            stat["iv"] = float("inf")
            stat["t_iv"] = float("inf")
        t_good = np.maximum(stat["good"].sum(), 0.5)
        t_bad = np.maximum(stat["bad"].sum(), 0.5)
        stat["woe2"] = stat.apply(_bucket_woe, axis=1) + np.log(t_good / t_bad)
        stat["iv2"] = (stat["bad"] / t_bad - stat["good"] / t_good) * stat["woe2"]
    return stat


def get_missing_bin(indata=None, var=None, nalist=[None, np.nan]):
    if indata is None:
        missing_bin = pd.DataFrame(columns=["LL", "VAR"])
        missing_bin["LL"] = pd.to_numeric(missing_bin["LL"])
    else:
        if indata.loc[indata[var].isin(nalist), :].shape[0]:
            missing_bin = pd.DataFrame({"LL": [None], "VAR": [var]})
        else:
            missing_bin = pd.DataFrame(columns=["LL", "VAR"])
            missing_bin["LL"] = pd.to_numeric(missing_bin["LL"])
    return missing_bin


def cut_bin(indata, var, bin_num, grp_method="freq"):
    if grp_method == "freq":
        tmp1 = pd.qcut(indata[var].values, bin_num).categories
        tmp2 = [v.left for v in tmp1][: len(tmp1) - 1]
        tmp2 = pd.DataFrame({"LL": tmp2})
    elif grp_method == "distance":
        tmp1 = pd.cut(indata[var].values, bin_num).categories
        tmp2 = [v.left for v in tmp1][: len(tmp1) - 1]
        tmp2 = pd.DataFrame({"LL": tmp2})
        ds = value_fit(x=pd.Series(indata[var].values), predefined_bins=tmp2["LL"])
        cut = set(list(itertools.chain.from_iterable(ds["bin_range"].str.replace(r"\(|\)|\[|\]", "", regex=True).str.split(",").tolist())))
        cut = [float(x) for x in cut]
        tmp2 = tmp2.loc[tmp2["LL"].isin(cut)].reset_index(drop=True)
    else:
        tmp1 = sorted(pd.unique(indata[var]))
        tmp2 = pd.DataFrame({"LL": tmp1[: len(tmp1) - 1]})
    return tmp2


def check_cutoff(indata, var, yname, bin_pct=0.05, diff=0.2):
    _tmp = indata.copy()
    _tmp["_count"] = 1
    shape = _tmp[["_count", var, yname]].groupby([yname, var])["_count"].count().reset_index()
    iv = _calc_woe(df=_tmp, x=var, y=yname).reset_index()
    iv[var] = pd.to_numeric(iv[var])
    iv = iv.sort_values(by=[var]).reset_index(drop=True)
    if iv.shape[0] > 1:
        flag_diff = True
        list_ = iv["woe2"].tolist()
        for item in enumerate(list_):
            if item[0] < len(list_) - 1:
                if (list_[item[0]] == 0) or (abs((list_[item[0]] - list_[item[0] + 1]) / list_[item[0]]) < diff):
                    flag_diff = False
                    break
        if not flag_diff:
            return False
    min_value = min(_tmp.groupby([var])[yname].count() / _tmp.shape[0])
    flag = (shape.loc[shape[yname] == 0, :].shape[0] == shape.loc[shape[yname] == 1, :].shape[0]) and (min_value >= bin_pct) and (shape.shape[0] >= 4)
    if flag:
        shape1 = _tmp.loc[_tmp[yname] == 1][["_count", var, yname]].groupby([yname, var])["_count"].count().reset_index().rename(columns={"_count": "bad"})
        shape2 = _tmp[["_count", var, yname]].groupby([var])["_count"].count().reset_index().rename(columns={"_count": "tot"})
        shape = pd.merge(shape1, shape2, on=var, how="outer")
        shape["rate"] = shape["bad"] / shape["tot"]
        shape = shape.sort_values(by=[var]).reset_index(drop=True)
        tmp1 = shape["rate"].diff()
        flag = abs(tmp1.sum()) == tmp1.abs().sum()
    return flag


class WoEbyValue:
    @staticmethod
    def cal_key(indata, testing, var, cutoff, yname, bin_pct, diff, key="auc"):
        tmp1 = indata.copy()
        cutoff = sorted(cutoff)
        if len(set(cutoff)) != len(cutoff):
            return -1
        cuts1 = pd.cut(tmp1[var], right=True, bins=[float("-inf")] + cutoff + [float("inf")], labels=np.arange(len(cutoff) + 1))
        tmp1["_var"] = np.where(cuts1.isnull(), "-9", cuts1.astype(str))
        flag = (len(set(np.arange(len(cutoff) + 1)) - set(pd.unique(cuts1))) == 0) and check_cutoff(indata=tmp1, var="_var", bin_pct=bin_pct, yname=yname, diff=diff)
        if testing is not None:
            for tmp2 in list(testing):
                cuts2 = pd.cut(tmp2[var], right=True, bins=[float("-inf")] + cutoff + [float("inf")], labels=np.arange(len(cutoff) + 1))
                tmp2["_var"] = np.where(cuts2.isnull(), "-9", cuts2.astype(str))
                flag2 = (len(set(np.arange(len(cutoff) + 1)) - set(pd.unique(cuts2))) == 0) and check_cutoff(indata=tmp2, var="_var", bin_pct=bin_pct, yname=yname, diff=diff)
                flag = flag and flag2
        if not flag:
            return -1
        if key.upper() == "AUC":
            tmp1["_var"] = pd.to_numeric(tmp1["_var"])
            fpr, tpr, _ = metrics.roc_curve(tmp1[yname], tmp1["_var"])
            return abs(0.5 - metrics.auc(fpr, tpr)) + 0.5
        if key.upper() == "KS":
            return ks_2samp(tmp1.loc[tmp1[yname] == 1, "_var"].values, tmp1.loc[tmp1[yname] == 0, "_var"].values).statistic
        return sum(_calc_woe(df=tmp1, x="_var", y=yname)["iv"])

    @staticmethod
    def get_bin(indata, testing, var, grp_method, yname, bin_pct, raw_bin=50, diff=0, value_type="KS"):
        initial_var_bin = cut_bin(indata=indata, var=var, bin_num=raw_bin, grp_method=grp_method)
        final_cut = []
        while True:
            max_value1 = []
            for x in initial_var_bin["LL"]:
                max_value1.append(
                    (
                        WoEbyValue.cal_key(
                            indata=indata,
                            testing=testing,
                            var=var,
                            cutoff=[x] + final_cut,
                            yname=yname,
                            bin_pct=bin_pct,
                            diff=diff,
                            key=value_type.upper(),
                        ),
                        x,
                    )
                )
            max_value2 = sorted(filter(lambda x: x[0] > 0, max_value1), reverse=True)
            if len(max_value2) == 0:
                break
            final_cut = final_cut + [max_value2[0][1]]
        var_bin_df = pd.DataFrame({"LL": final_cut})
        var_bin_df["VAR"] = var
        return var_bin_df[["LL", "VAR"]].sort_values(by=["LL"]).reset_index(drop=True)

    @staticmethod
    def get_varbin(indata, testing, var, yname, type_rawbin, max_rawbin, bin_pct, value_type, diff=0):
        des = "{value_type}_var({var})grpmethod({grp_method})".format(value_type=value_type, var=var, grp_method=type_rawbin)
        missing_bin = get_missing_bin(indata=indata, var=var, nalist=[None, np.nan])
        try:
            tmp = indata.loc[indata[var].notnull(), :].reset_index(drop=True)
            var_bin_df = WoEbyValue.get_bin(
                indata=tmp,
                testing=testing,
                var=var,
                grp_method=type_rawbin,
                yname=yname,
                raw_bin=max_rawbin,
                bin_pct=bin_pct,
                diff=diff,
                value_type=value_type,
            )
        except Exception:
            var_bin_df = pd.DataFrame(columns=["VAR", "LL"])
        if not var_bin_df.empty:
            var_bin_df = pd.concat([missing_bin, var_bin_df], ignore_index=True)
        var_bin_df["IDVAR"] = des
        return var_bin_df


def var_bin(indata, testing, var, yname, bintype, bin_param, outlier, char_flag=False):
    tmp = indata.copy()
    if char_flag or (len(pd.unique(indata[var].dropna())) < 3):
        return_ds = pd.DataFrame({"LL": sorted(pd.unique(indata[var].dropna()))})
        return_ds = return_ds.sort_values(["LL"]).reset_index(drop=True)
        return_ds["VAR"] = var
        return_ds["IDVAR"] = "{bintype}_var({var})grpmethod({type_rawbin})".format(
            bintype=bintype,
            var=var,
            type_rawbin=bin_param.get("type_rawbin"),
        )
    else:
        if outlier == "99%":
            _1cut, _99cut = np.percentile(list(filter(lambda x: pd.notnull(x), tmp[var])), [1, 99])
            tmp[var] = np.where(tmp[var] >= _99cut, _99cut, tmp[var])
            tmp[var] = np.where(tmp[var] <= _1cut, _1cut, tmp[var])
        return_ds = WoEbyValue.get_varbin(
            indata=tmp,
            testing=testing,
            var=var,
            yname=yname,
            value_type=bintype,
            type_rawbin=bin_param.get("type_rawbin"),
            max_rawbin=bin_param.get("max_rawbin"),
            bin_pct=bin_param.get("bin_pct"),
            diff=bin_param.get("diff"),
        )
    return return_ds.sort_values(by=["VAR", "IDVAR", "LL"])


def transform(indata=None, woedata=None, var="VAR", bin_range="bin_range", woevar="woe", suffix="_woe"):
    finalds = indata.reset_index(drop=True).copy()
    for var_value in woedata[var].unique():
        woeds = woedata.loc[(woedata[var] == var_value), [woevar, bin_range, "_labels_"]]
        cutoff_value = set(list(itertools.chain.from_iterable(woeds[bin_range].str.replace(r"\(|\)|\[|\]", "", regex=True).str.split(",").tolist())))
        cutoff_value = [float(name) for name in cutoff_value if name.strip() not in ["-inf", "inf", "missing"]]
        ds = value_fit(x=pd.Series(indata[var_value].values), predefined_bins=sorted(cutoff_value))
        finalds[var_value + suffix] = pd.merge(ds[["_labels_", "bin_range"]], woeds, on="_labels_", how="left")[[woevar]]
    return finalds


def calc_woe_details(indata, yname, bindata, var_value, ll="LL", var="VAR"):
    cut = bindata.loc[(bindata[var] == var_value), ll].tolist()
    cut = list(filter(lambda x: pd.notnull(x), cut))
    ds = value_fit(x=pd.Series(indata[var_value].values), predefined_bins=cut)
    ds["Y"] = indata[yname].values
    stats = _calc_woe(df=ds, x="_labels_", y="Y").reset_index().rename(columns={"index": "_labels_"})
    if stats.empty:
        return None, None

    rangeds = get_range(predefined_bins=cut)
    detail_df = pd.merge(stats, rangeds, on="_labels_", how="left")
    detail_df["bin_range"] = detail_df["bin_range"].fillna("missing")
    detail_df["bin_range2"] = detail_df["bin_range2"].fillna("missing")
    detail_df[var] = var_value
    detail_df["woe"] = detail_df["woe2"]
    detail_df["pct"] = detail_df["obs"] / indata.shape[0]
    detail_df["count"] = detail_df["obs"]
    detail_df["count_good"] = detail_df["good"]
    detail_df["count_bad"] = detail_df["bad"]
    detail_df["bad_rate"] = detail_df["overdue_rate"]

    transform_df = detail_df[[var, "_labels_", "bin_range", "woe"]].copy()
    detail_df = detail_df[
        [
            var,
            "_labels_",
            "bin_range",
            "bin_range2",
            "count",
            "count_good",
            "count_bad",
            "bad_rate",
            "woe",
            "pct",
            "iv2",
        ]
    ].rename(columns={var: "VAR"})
    detail_df = detail_df.rename(columns={"iv2": "iv"})
    return detail_df, transform_df.rename(columns={var: "VAR"})
