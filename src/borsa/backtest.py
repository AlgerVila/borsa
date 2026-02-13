from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from borsa.config import AppConfig
from borsa.factors import build_factor_frame
from borsa.scoring import build_factor_scores, compute_final_scores, select_top_n


@dataclass
class BacktestResult:
    performance: pd.DataFrame
    metrics: pd.DataFrame


def _price_column(df_prices: pd.DataFrame) -> str:
    if "adj_close" in df_prices.columns:
        return "adj_close"
    return "close"


def _price_matrix(df_prices: pd.DataFrame) -> pd.DataFrame:
    price_col = _price_column(df_prices)
    px = df_prices[["date", "ticker", price_col]].copy()
    px["date"] = pd.to_datetime(px["date"], utc=False)
    wide = px.pivot(index="date", columns="ticker", values=price_col).sort_index()
    return wide


def _monthly_rebalance_dates(index: pd.DatetimeIndex, start: date, end: date) -> list[pd.Timestamp]:
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if idx.empty:
        return []
    by_month = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    return list(by_month.sort_values())


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _metrics(name: str, returns: pd.Series, freq: int = 12) -> dict:
    r = returns.dropna()
    if r.empty:
        return {
            "name": name,
            "periods": 0,
            "cagr": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    n = len(r)
    total = float((1.0 + r).prod())
    cagr = total ** (freq / n) - 1.0
    vol = float(r.std(ddof=1) * np.sqrt(freq)) if n > 1 else np.nan
    mean = float(r.mean() * freq)
    sharpe = mean / vol if vol and vol > 0 else np.nan
    return {
        "name": name,
        "periods": n,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(r),
    }


def run_monthly_backtest(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    cfg: AppConfig,
    start: date,
    end: date,
    top_n: int = 10,
    cost_bps: float = 10.0,
) -> BacktestResult:
    if prices.empty:
        return BacktestResult(performance=pd.DataFrame(), metrics=pd.DataFrame())

    full_prices = prices.copy()
    full_prices["date"] = pd.to_datetime(full_prices["date"], utc=False)
    matrix = _price_matrix(full_prices)
    bench_matrix = _price_matrix(benchmark_prices) if not benchmark_prices.empty else pd.DataFrame()
    rebalances = _monthly_rebalance_dates(matrix.index, start=start, end=end)
    if len(rebalances) < 2:
        return BacktestResult(performance=pd.DataFrame(), metrics=pd.DataFrame())

    records: list[dict] = []
    prev_holdings: set[str] | None = None

    for i in range(len(rebalances) - 1):
        reb_date = rebalances[i]
        next_date = rebalances[i + 1]

        hist = full_prices[full_prices["date"] <= reb_date].copy()
        factors = build_factor_frame(df_price=hist, df_fund=fundamentals, cfg=cfg)
        scored = compute_final_scores(build_factor_scores(factors=factors, cfg=cfg))
        top = select_top_n(scored, n=top_n)
        holdings = set(top["ticker"].tolist())
        if not holdings:
            continue

        p0 = matrix.loc[reb_date, list(holdings)].dropna()
        p1 = matrix.loc[next_date, list(holdings)].dropna()
        common = p0.index.intersection(p1.index)
        if len(common) == 0:
            continue
        period_ret = (p1.loc[common] / p0.loc[common] - 1.0).mean()

        turnover = 1.0 if prev_holdings is None else 1.0 - (len(holdings & prev_holdings) / max(len(holdings), 1))
        cost = turnover * (cost_bps / 10_000.0)
        net_ret = float(period_ret - cost)

        bench_ret = np.nan
        if "QQQ" in bench_matrix.columns and reb_date in bench_matrix.index and next_date in bench_matrix.index:
            b0 = bench_matrix.loc[reb_date, "QQQ"]
            b1 = bench_matrix.loc[next_date, "QQQ"]
            if pd.notna(b0) and pd.notna(b1) and b0 > 0:
                bench_ret = float(b1 / b0 - 1.0)

        records.append(
            {
                "rebalance_date": reb_date.date().isoformat(),
                "next_rebalance_date": next_date.date().isoformat(),
                "strategy_return": net_ret,
                "benchmark_return": bench_ret,
                "turnover": turnover,
                "holdings_count": len(common),
            }
        )
        prev_holdings = holdings

    perf = pd.DataFrame(records)
    if perf.empty:
        return BacktestResult(performance=perf, metrics=pd.DataFrame())

    strategy = perf["strategy_return"]
    bench = perf["benchmark_return"]
    metrics = pd.DataFrame(
        [
            _metrics("strategy_top10", strategy),
            _metrics("benchmark_qqq", bench),
            {
                "name": "turnover",
                "periods": int(perf["turnover"].notna().sum()),
                "cagr": np.nan,
                "volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "average_turnover": float(perf["turnover"].mean()),
            },
        ]
    )
    return BacktestResult(performance=perf, metrics=metrics)
