from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from borsa.config import cache_file_for


def _period_tag(period: str) -> str:
    return period.replace(" ", "").replace("/", "_")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def fetch_price_history(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df.empty:
        raise ValueError("Yahoo returned empty price history")

    if isinstance(df.columns, pd.MultiIndex):
        stacked = (
            df.stack(level=0, future_stack=True)
            .rename_axis(index=["date", "ticker"])
            .reset_index()
        )
        stacked.columns = [str(c).lower().replace(" ", "_") for c in stacked.columns]
        return stacked

    # Single ticker fallback shape.
    out = df.reset_index()
    out["ticker"] = tickers[0]
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def _fetch_single_fundamental(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    if not isinstance(info, dict) or not info:
        raise ValueError(f"No fundamentals for {ticker}")
    info["ticker"] = ticker
    return info


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker in tickers:
        try:
            rows.append(_fetch_single_fundamental(ticker))
        except Exception:
            rows.append({"ticker": ticker})
    return pd.DataFrame(rows)


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_or_fetch_prices(
    tickers: list[str],
    as_of: date,
    cache_dir: Path,
    period: str = "2y",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, Path]:
    path = cache_file_for(f"prices_{_period_tag(period)}", as_of, cache_dir)
    if use_cache and path.exists():
        return _load_parquet(path), path

    prices = fetch_price_history(tickers=tickers, period=period)
    _save_parquet(prices, path)
    return prices, path


def get_or_fetch_fundamentals(
    tickers: list[str],
    as_of: date,
    cache_dir: Path,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, Path]:
    path = cache_file_for("fundamentals", as_of, cache_dir)
    if use_cache and path.exists():
        return _load_parquet(path), path

    fundamentals = fetch_fundamentals(tickers=tickers)
    _save_parquet(fundamentals, path)
    return fundamentals, path


def get_or_fetch_benchmark(
    ticker: str,
    as_of: date,
    cache_dir: Path,
    period: str = "10y",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, Path]:
    path = cache_file_for(f"benchmark_{ticker}_{_period_tag(period)}", as_of, cache_dir)
    if use_cache and path.exists():
        return _load_parquet(path), path

    bench = fetch_price_history([ticker], period=period)
    _save_parquet(bench, path)
    return bench, path
