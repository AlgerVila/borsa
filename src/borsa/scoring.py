from __future__ import annotations

import numpy as np
import pandas as pd

from borsa.config import AppConfig

FACTOR_SPECS: list[tuple[str, float, bool]] = [
    # Long-term value profile:
    # valuation + quality dominate, short-term momentum is reduced.
    ("trailingPE", 15.0, False),
    ("priceToSalesTrailing12Months", 10.0, False),
    ("pegRatio", 7.0, False),
    ("fcf_yield", 10.0, True),
    ("returnOnEquity", 12.0, True),
    ("returnOnAssets", 8.0, True),
    ("operatingMargins", 8.0, True),
    ("grossMargins", 5.0, True),
    ("revenueGrowth", 7.0, True),
    ("earningsGrowth", 5.0, True),
    ("debtToEquity", 5.0, False),
    ("mom_12_1", 5.0, True),
    ("beta", 2.0, False),
    ("vol_63d", 0.5, False),
    ("max_drawdown_252d", 0.5, False),
]

FACTOR_GROUPS: dict[str, list[str]] = {
    "value_score": ["trailingPE", "priceToSalesTrailing12Months", "pegRatio", "fcf_yield"],
    "quality_score": ["returnOnEquity", "returnOnAssets", "operatingMargins", "grossMargins"],
    "growth_score": ["revenueGrowth", "earningsGrowth"],
    "stability_score": ["debtToEquity", "beta", "vol_63d", "max_drawdown_252d"],
    "momentum_score": ["mom_12_1"],
}


def _winsorize(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 3:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def percentile_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    ranks = s.rank(method="average", pct=True) * 100.0
    if not higher_is_better:
        ranks = 100.0 - ranks
    return ranks


def _factor_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name, _, _ in FACTOR_SPECS if name in frame.columns]


def build_factor_scores(factors: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    if factors.empty:
        return pd.DataFrame()

    out = factors.copy()
    cols = _factor_columns(out)

    for col, _, higher_is_better in FACTOR_SPECS:
        if col not in out.columns:
            continue
        out[col] = _winsorize(out[col])
        out[col] = percentile_rank(out[col], higher_is_better=higher_is_better)

    out["missing_count"] = out[cols].isna().sum(axis=1)
    max_missing = int(np.floor(len(cols) * cfg.filters.max_missing_factor_ratio))
    out = out[out["missing_count"] <= max_missing].copy()

    out[cols] = out[cols].fillna(50.0)
    out["missing_penalty"] = (-2.0 * out["missing_count"]).clip(lower=-10.0, upper=0.0)
    return out


def compute_final_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    out = scores.copy()
    weight_map = {name: weight for name, weight, _ in FACTOR_SPECS}
    weighted_sum = 0.0
    for col, weight, _ in FACTOR_SPECS:
        if col in out.columns:
            weighted_sum += out[col] * weight

    out["score"] = weighted_sum / 100.0
    for group_name, factors in FACTOR_GROUPS.items():
        present = [f for f in factors if f in out.columns]
        if not present:
            continue
        group_weight = sum(weight_map[f] for f in present)
        if group_weight <= 0:
            continue
        group_num = 0.0
        for f in present:
            group_num += out[f] * weight_map[f]
        out[group_name] = group_num / group_weight

    out["final_score"] = out["score"] + out["missing_penalty"]
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def select_top_n(final_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if final_df.empty:
        return final_df
    return final_df.head(n).copy()
