from __future__ import annotations

from datetime import date
import math
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from borsa.config import cache_file_for


def _period_tag(period: str) -> str:
    return period.replace(" ", "").replace("/", "_")


def _to_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _first_numeric(mapping: dict[str, object], keys: list[str]) -> float | None:
    for key in keys:
        if key not in mapping:
            continue
        val = _to_float(mapping.get(key))
        if val is not None:
            return val
    return None


def _canonicalize_fundamental_fields(info: dict[str, object], fast_info: dict[str, object]) -> dict[str, object]:
    merged = dict(info)

    # Keep a stable market cap field when fast_info has it but info does not.
    market_cap = _first_numeric(merged, ["marketCap"])
    if market_cap is None:
        market_cap = _first_numeric(fast_info, ["marketCap", "market_cap"])
        if market_cap is not None:
            merged["marketCap"] = market_cap

    price = _first_numeric(merged, ["currentPrice", "regularMarketPrice", "previousClose"])
    if price is None:
        price = _first_numeric(fast_info, ["lastPrice", "last_price", "regularMarketPrice"])
        if price is not None:
            merged["currentPrice"] = price

    trailing_pe = _first_numeric(merged, ["trailingPE", "trailingPe", "trailing_pe"])
    if trailing_pe is None:
        trailing_pe = _first_numeric(fast_info, ["trailingPE", "trailing_pe", "peRatio", "pe_ratio"])
    if trailing_pe is None and price is not None:
        trailing_eps = _first_numeric(merged, ["trailingEps", "epsTrailingTwelveMonths", "trailing_eps"])
        if trailing_eps is not None and trailing_eps > 0:
            trailing_pe = price / trailing_eps
    if trailing_pe is not None and trailing_pe > 0:
        merged["trailingPE"] = trailing_pe

    ps = _first_numeric(
        merged,
        [
            "priceToSalesTrailing12Months",
            "priceToSalesTrailingTwelveMonths",
            "priceToSales",
            "price_to_sales",
            "psRatio",
        ],
    )
    if ps is None:
        ps = _first_numeric(fast_info, ["priceToSalesTrailing12Months", "price_to_sales", "psRatio"])
    if ps is None:
        total_revenue = _first_numeric(merged, ["totalRevenue", "revenue"])
        if market_cap is not None and total_revenue is not None and total_revenue > 0:
            ps = market_cap / total_revenue
    if ps is None and price is not None:
        revenue_per_share = _first_numeric(merged, ["revenuePerShare"])
        if revenue_per_share is not None and revenue_per_share > 0:
            ps = price / revenue_per_share
    if ps is not None and ps > 0:
        merged["priceToSalesTrailing12Months"] = ps

    peg_ratio = _first_numeric(merged, ["pegRatio", "trailingPegRatio", "peg_ratio"])
    if peg_ratio is not None and peg_ratio > 0:
        merged["pegRatio"] = peg_ratio

    return merged


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
    ticker_obj = yf.Ticker(ticker)
    info: dict[str, object] = {}
    try:
        fetched = ticker_obj.info
        if isinstance(fetched, dict):
            info = fetched
    except Exception:
        info = {}

    if not info:
        try:
            fetched = ticker_obj.get_info()
            if isinstance(fetched, dict):
                info = fetched
        except Exception:
            info = {}

    fast_info: dict[str, object] = {}
    try:
        fi = ticker_obj.fast_info
        getter = getattr(fi, "get", None)
        if callable(getter):
            for key in [
                "marketCap",
                "market_cap",
                "lastPrice",
                "last_price",
                "regularMarketPrice",
                "trailingPE",
                "trailing_pe",
                "peRatio",
                "pe_ratio",
                "priceToSalesTrailing12Months",
                "price_to_sales",
            ]:
                val = getter(key)
                if val is not None:
                    fast_info[key] = val
        elif isinstance(fi, dict):
            fast_info = fi
    except Exception:
        fast_info = {}

    if not info and not fast_info:
        raise ValueError(f"No fundamentals for {ticker}")

    normalized = _canonicalize_fundamental_fields(info=info, fast_info=fast_info)
    normalized["ticker"] = ticker
    return normalized


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
