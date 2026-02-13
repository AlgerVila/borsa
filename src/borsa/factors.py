from __future__ import annotations

import numpy as np
import pandas as pd

from borsa.config import AppConfig


QUALITY_VALUE_COLUMNS = [
    "returnOnEquity",
    "returnOnAssets",
    "operatingMargins",
    "grossMargins",
    "revenueGrowth",
    "earningsGrowth",
    "trailingPE",
    "priceToSalesTrailing12Months",
    "pegRatio",
    "debtToEquity",
    "freeCashflow",
    "beta",
    "marketCap",
    "netIncomeToCommon",
]


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_ratio_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # Some feeds provide percent points (e.g. 25 for 25%), convert to decimal.
    pct_like = s.abs().between(2.0, 100.0, inclusive="both")
    s = s.where(~pct_like, s / 100.0)
    # Drop pathological outliers from unstable balance-sheet denominators.
    return s.where(s.abs() <= 5.0)


def _normalize_debt_to_equity(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # Yahoo often reports D/E in percent units (e.g. 120 => 1.2x).
    pct_like = s.abs() > 5.0
    s = s.where(~pct_like, s / 100.0)
    return s.where(s >= 0)


def _price_column(df_prices: pd.DataFrame) -> str:
    if "adj_close" in df_prices.columns:
        return "adj_close"
    return "close"


def _prepare_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    if df_prices.empty:
        return df_prices.copy()
    out = df_prices.copy()
    out["date"] = pd.to_datetime(out["date"], utc=False)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def _average_daily_dollar_volume(df_prices: pd.DataFrame, window: int = 20) -> pd.Series:
    price_col = _price_column(df_prices)
    px = df_prices[["ticker", "date", price_col, "volume"]].copy()
    px["dollar_volume"] = px[price_col] * px["volume"]
    adv = (
        px.groupby("ticker", sort=False)["dollar_volume"]
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    px["adv"] = adv
    return px.groupby("ticker", sort=False)["adv"].last()


def _counts_by_ticker(df_prices: pd.DataFrame) -> pd.Series:
    return df_prices.groupby("ticker", sort=False)["date"].count()


def filter_diagnostics(df_fund: pd.DataFrame, df_price: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    prices = _prepare_prices(df_price)
    if prices.empty or "ticker" not in df_fund.columns:
        return pd.DataFrame(columns=["step", "count"])

    funds = df_fund.drop_duplicates(subset=["ticker"]).set_index("ticker")
    all_tickers = set(prices["ticker"].dropna().unique())
    step_sets: list[tuple[str, set[str]]] = [("universe", all_tickers)]

    counts = _counts_by_ticker(prices)
    enough_history = set(counts[counts >= cfg.lookbacks.trading_days_year].index)
    step_sets.append(("history>=252d", enough_history))

    market_cap = pd.to_numeric(funds.get("marketCap"), errors="coerce")
    market_cap_ok = set(funds.index[(market_cap >= cfg.filters.min_market_cap).fillna(False)])
    step_sets.append(("market_cap", enough_history & market_cap_ok))

    adv = _average_daily_dollar_volume(prices)
    adv_ok = set(adv.index[(adv >= cfg.filters.min_avg_daily_dollar_volume).fillna(False)])
    step_sets.append(("liquidity", (enough_history & market_cap_ok) & adv_ok))

    current = (enough_history & market_cap_ok) & adv_ok
    if cfg.filters.require_positive_net_income and "netIncomeToCommon" in funds.columns:
        ni = pd.to_numeric(funds["netIncomeToCommon"], errors="coerce")
        ni_allowed = set(funds.index[(ni > 0).fillna(False) | ni.isna()])
        current = current & ni_allowed
    step_sets.append(("net_income", current))

    if cfg.filters.require_pe_bounds and "trailingPE" in funds.columns:
        pe = pd.to_numeric(funds["trailingPE"], errors="coerce")
        pe_ok = set(funds.index[((pe > cfg.filters.min_trailing_pe) & (pe <= cfg.filters.max_trailing_pe)).fillna(False)])
        current = current & pe_ok
    step_sets.append(("pe_cap", current))

    if cfg.filters.require_ps_cap and "priceToSalesTrailing12Months" in funds.columns:
        ps = pd.to_numeric(funds["priceToSalesTrailing12Months"], errors="coerce")
        ps_ok = set(funds.index[((ps > 0) & (ps <= cfg.filters.max_price_to_sales)).fillna(False)])
        current = current & ps_ok
    step_sets.append(("ps_cap", current))

    peg_col = _first_existing_column(funds, ["pegRatio", "trailingPegRatio"])
    if cfg.filters.require_peg_cap and peg_col is not None:
        peg = pd.to_numeric(funds[peg_col], errors="coerce")
        peg_ok = set(funds.index[((peg > 0) & (peg <= cfg.filters.max_peg_ratio)).fillna(False)])
        current = current & peg_ok
    step_sets.append(("peg_cap", current))

    return pd.DataFrame([{"step": name, "count": len(tickers)} for name, tickers in step_sets])


def apply_hard_filters(df_fund: pd.DataFrame, df_price: pd.DataFrame, cfg: AppConfig) -> pd.Index:
    prices = _prepare_prices(df_price)
    if prices.empty:
        return pd.Index([])

    required_cols = {"ticker", "marketCap"}
    if not required_cols.issubset(df_fund.columns):
        df_fund = df_fund.copy()
        if "ticker" not in df_fund.columns:
            return pd.Index([])
        if "marketCap" not in df_fund.columns:
            df_fund["marketCap"] = np.nan

    funds = df_fund.drop_duplicates(subset=["ticker"]).set_index("ticker")
    counts = _counts_by_ticker(prices)
    adv = _average_daily_dollar_volume(prices)

    valid = counts[counts >= cfg.lookbacks.trading_days_year].index
    market_cap_ok = funds["marketCap"] >= cfg.filters.min_market_cap
    adv_ok = adv >= cfg.filters.min_avg_daily_dollar_volume

    keep = set(valid) & set(funds.index[market_cap_ok.fillna(False)]) & set(adv.index[adv_ok.fillna(False)])

    if cfg.filters.require_positive_net_income and "netIncomeToCommon" in funds.columns:
        ni_ok = funds["netIncomeToCommon"] > 0
        ni_unknown = funds["netIncomeToCommon"].isna()
        ni_allowed = funds.index[ni_ok.fillna(False) | ni_unknown]
        keep &= set(ni_allowed)

    if cfg.filters.require_pe_bounds and "trailingPE" in funds.columns:
        pe = pd.to_numeric(funds["trailingPE"], errors="coerce")
        pe_ok = (pe > cfg.filters.min_trailing_pe) & (pe <= cfg.filters.max_trailing_pe)
        keep &= set(funds.index[pe_ok.fillna(False)])

    if cfg.filters.require_ps_cap and "priceToSalesTrailing12Months" in funds.columns:
        ps = pd.to_numeric(funds["priceToSalesTrailing12Months"], errors="coerce")
        ps_ok = (ps > 0) & (ps <= cfg.filters.max_price_to_sales)
        keep &= set(funds.index[ps_ok.fillna(False)])

    peg_col = _first_existing_column(funds, ["pegRatio", "trailingPegRatio"])
    if cfg.filters.require_peg_cap and peg_col is not None:
        peg = pd.to_numeric(funds[peg_col], errors="coerce")
        peg_ok = (peg > 0) & (peg <= cfg.filters.max_peg_ratio)
        keep &= set(funds.index[peg_ok.fillna(False)])

    return pd.Index(sorted(keep))


def compute_momentum_factors(df_price: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    prices = _prepare_prices(df_price)
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "mom_12_1", "mom_6_1", "mom_3_1"])

    price_col = _price_column(prices)
    g = prices.groupby("ticker", sort=False)[price_col]

    p_t = g.shift(0)
    p_21 = g.shift(cfg.lookbacks.skip_recent_days)
    p_63 = g.shift(cfg.lookbacks.momentum_short)
    p_126 = g.shift(cfg.lookbacks.momentum_mid)
    p_252 = g.shift(cfg.lookbacks.momentum_long)

    prices["mom_12_1"] = p_21 / p_252 - 1.0
    prices["mom_6_1"] = p_21 / p_126 - 1.0
    prices["mom_3_1"] = p_21 / p_63 - 1.0
    prices["_p_t"] = p_t

    last = prices.groupby("ticker", sort=False).tail(1)
    return last[["ticker", "mom_12_1", "mom_6_1", "mom_3_1"]].reset_index(drop=True)


def _max_drawdown_last_window(series: pd.Series, window: int) -> float:
    s = series.dropna()
    if len(s) < window:
        return np.nan
    s = s.iloc[-window:]
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return float(abs(dd.min()))


def compute_risk_factors(df_price: pd.DataFrame, df_fund: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    prices = _prepare_prices(df_price)
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "beta", "vol_63d", "max_drawdown_252d"])

    price_col = _price_column(prices)
    prices["ret_1d"] = prices.groupby("ticker", sort=False)[price_col].pct_change(fill_method=None)
    vol = (
        prices.groupby("ticker", sort=False)["ret_1d"]
        .rolling(cfg.lookbacks.risk_vol_window, min_periods=cfg.lookbacks.risk_vol_window)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(cfg.lookbacks.trading_days_year)
    )
    prices["vol_63d"] = vol

    vol_last = prices.groupby("ticker", sort=False).tail(1)[["ticker", "vol_63d"]]
    mdd = (
        prices.groupby("ticker", sort=False)[price_col]
        .apply(lambda s: _max_drawdown_last_window(s, cfg.lookbacks.trading_days_year))
        .rename("max_drawdown_252d")
        .reset_index()
    )
    out = vol_last.merge(mdd, on="ticker", how="left")

    if "ticker" in df_fund.columns and "beta" in df_fund.columns:
        beta = df_fund[["ticker", "beta"]].drop_duplicates(subset=["ticker"])
        out = out.merge(beta, on="ticker", how="left")
    else:
        out["beta"] = np.nan

    return out[["ticker", "beta", "vol_63d", "max_drawdown_252d"]]


def compute_quality_value_factors(df_fund: pd.DataFrame) -> pd.DataFrame:
    if df_fund.empty or "ticker" not in df_fund.columns:
        return pd.DataFrame(
            columns=[
                "ticker",
                "returnOnEquity",
                "returnOnAssets",
                "operatingMargins",
                "grossMargins",
                "revenueGrowth",
                "earningsGrowth",
                "trailingPE",
                "priceToSalesTrailing12Months",
                "pegRatio",
                "debtToEquity",
                "fcf_yield",
            ]
        )

    peg_col = _first_existing_column(df_fund, ["pegRatio", "trailingPegRatio"])
    work = df_fund.copy()
    if peg_col is not None and peg_col != "pegRatio":
        work["pegRatio"] = work[peg_col]

    cols = [c for c in QUALITY_VALUE_COLUMNS if c in work.columns]
    out = work[["ticker", *cols]].drop_duplicates(subset=["ticker"]).copy()
    for missing in set(QUALITY_VALUE_COLUMNS) - set(cols):
        out[missing] = np.nan

    for col in [
        "returnOnEquity",
        "returnOnAssets",
        "operatingMargins",
        "grossMargins",
        "revenueGrowth",
        "earningsGrowth",
    ]:
        out[col] = _normalize_ratio_series(out[col])

    out["debtToEquity"] = _normalize_debt_to_equity(out["debtToEquity"])
    out["trailingPE"] = pd.to_numeric(out["trailingPE"], errors="coerce")
    out["priceToSalesTrailing12Months"] = pd.to_numeric(out["priceToSalesTrailing12Months"], errors="coerce")
    out["pegRatio"] = pd.to_numeric(out["pegRatio"], errors="coerce")
    out["marketCap"] = pd.to_numeric(out["marketCap"], errors="coerce")
    out["freeCashflow"] = pd.to_numeric(out["freeCashflow"], errors="coerce")
    out["fcf_yield"] = np.where(
        out["marketCap"] > 0,
        out["freeCashflow"] / out["marketCap"],
        np.nan,
    )

    return out[
        [
            "ticker",
            "returnOnEquity",
            "returnOnAssets",
            "operatingMargins",
            "grossMargins",
            "revenueGrowth",
            "earningsGrowth",
            "trailingPE",
            "priceToSalesTrailing12Months",
            "pegRatio",
            "debtToEquity",
            "fcf_yield",
        ]
    ]


def build_factor_frame(df_price: pd.DataFrame, df_fund: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    keep = apply_hard_filters(df_fund=df_fund, df_price=df_price, cfg=cfg)
    if keep.empty:
        return pd.DataFrame()

    mom = compute_momentum_factors(df_price, cfg)
    risk = compute_risk_factors(df_price, df_fund, cfg)
    qv = compute_quality_value_factors(df_fund)

    factors = mom.merge(risk, on="ticker", how="outer").merge(qv, on="ticker", how="outer")
    factors = factors[factors["ticker"].isin(set(keep))].reset_index(drop=True)
    return factors
