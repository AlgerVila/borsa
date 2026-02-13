from __future__ import annotations

from datetime import date

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

WIKI_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Fallback snapshot list in case upstream source is unavailable.
FALLBACK_NASDAQ100 = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "CDNS", "CEG",
    "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSX", "CTAS", "CTSH", "DASH",
    "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS",
    "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KHC",
    "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META", "MNST",
    "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY",
    "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN", "ROP", "ROST",
    "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX",
    "WBA", "WDAY", "XEL", "ZS",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _download_wiki_table() -> pd.DataFrame:
    response = requests.get(WIKI_NDX_URL, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(response.text)

    for table in tables:
        cols = [str(col).strip().lower() for col in table.columns]
        if "ticker" in cols:
            return table
        if "symbol" in cols:
            return table

    raise ValueError("No table with ticker/symbol column found on Wikipedia page")


def normalize_for_yahoo(ticker: str) -> str:
    # Yahoo Finance uses '-' for share classes like BRK.B => BRK-B
    return ticker.strip().upper().replace(".", "-")


def get_nasdaq100_tickers(as_of: date | None = None) -> list[str]:
    _ = as_of  # Reserved for future point-in-time universe support.

    try:
        table = _download_wiki_table()
        column_map = {str(c).strip().lower(): c for c in table.columns}
        ticker_col = column_map.get("ticker") or column_map.get("symbol")
        if ticker_col is None:
            raise ValueError("Ticker column not found after loading table")
        tickers = sorted(
            {
                normalize_for_yahoo(str(item))
                for item in table[ticker_col].dropna().tolist()
                if str(item).strip()
            }
        )
        if len(tickers) < 90:
            raise ValueError(f"Unexpected ticker count from source: {len(tickers)}")
        return tickers
    except Exception:
        return sorted(normalize_for_yahoo(t) for t in FALLBACK_NASDAQ100)
