from datetime import date

import numpy as np
import pandas as pd

from borsa.backtest import run_monthly_backtest
from borsa.config import AppConfig


def _synthetic_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    rows = []
    for i, t in enumerate(tickers):
        base = 100 + (i * 5)
        trend = np.linspace(0, 25 + i, len(idx))
        series = base + trend
        for d, px in zip(idx, series):
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "adj_close": px,
                    "volume": 2_000_000 + i * 1000,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        rows.append(
            {
                "ticker": t,
                "marketCap": 10_000_000_000 + i * 100_000_000,
                "netIncomeToCommon": 500_000_000,
                "returnOnEquity": 0.10 + i * 0.01,
                "operatingMargins": 0.12 + i * 0.005,
                "revenueGrowth": 0.06 + i * 0.003,
                "trailingPE": 20 + i,
                "priceToSalesTrailing12Months": 4 + i * 0.2,
                "pegRatio": 1.5 + i * 0.05,
                "beta": 1.0 + i * 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_run_monthly_backtest_produces_metrics() -> None:
    tickers = [f"T{i:02d}" for i in range(1, 16)]
    prices = _synthetic_prices(tickers, "2022-01-01", "2025-12-31")
    funds = _synthetic_fundamentals(tickers)
    bench = _synthetic_prices(["QQQ"], "2022-01-01", "2025-12-31")

    cfg = AppConfig()
    result = run_monthly_backtest(
        prices=prices,
        fundamentals=funds,
        benchmark_prices=bench,
        cfg=cfg,
        start=date(2023, 1, 1),
        end=date(2025, 12, 31),
        top_n=10,
        cost_bps=10.0,
    )

    assert not result.performance.empty
    assert "strategy_return" in result.performance.columns
    assert "benchmark_return" in result.performance.columns
    assert not result.metrics.empty
    assert "name" in result.metrics.columns
