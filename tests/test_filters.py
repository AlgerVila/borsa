import pandas as pd

from borsa.config import AppConfig
from borsa.factors import apply_hard_filters


def test_hard_filter_enforces_trailing_pe_cap() -> None:
    cfg = AppConfig()
    cfg.filters.max_trailing_pe = 40.0

    dates = pd.bdate_range("2025-01-01", periods=260)
    price_rows = []
    for ticker in ["LOWPE", "HIGHPE"]:
        for d in dates:
            price_rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 500_000,
                }
            )
    prices = pd.DataFrame(price_rows)

    funds = pd.DataFrame(
        [
            {
                "ticker": "LOWPE",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPE": 25.0,
            },
            {
                "ticker": "HIGHPE",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPE": 120.0,
            },
        ]
    )

    keep = apply_hard_filters(df_fund=funds, df_price=prices, cfg=cfg)
    assert keep.tolist() == ["LOWPE"]


def test_hard_filter_enforces_ps_and_peg_caps() -> None:
    cfg = AppConfig()
    cfg.filters.max_price_to_sales = 10.0
    cfg.filters.max_peg_ratio = 2.5

    dates = pd.bdate_range("2025-01-01", periods=260)
    rows = []
    for ticker in ["CHEAP", "EXPENSIVE_PS", "EXPENSIVE_PEG"]:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 500_000,
                }
            )
    prices = pd.DataFrame(rows)

    funds = pd.DataFrame(
        [
            {
                "ticker": "CHEAP",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPE": 20.0,
                "priceToSalesTrailing12Months": 4.0,
                "trailingPegRatio": 1.5,
            },
            {
                "ticker": "EXPENSIVE_PS",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPE": 20.0,
                "priceToSalesTrailing12Months": 14.0,
                "trailingPegRatio": 1.5,
            },
            {
                "ticker": "EXPENSIVE_PEG",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPE": 20.0,
                "priceToSalesTrailing12Months": 4.0,
                "trailingPegRatio": 3.5,
            },
        ]
    )

    keep = apply_hard_filters(df_fund=funds, df_price=prices, cfg=cfg)
    assert keep.tolist() == ["CHEAP"]


def test_hard_filter_supports_pe_ps_alias_columns() -> None:
    cfg = AppConfig()
    cfg.filters.max_trailing_pe = 40.0
    cfg.filters.max_price_to_sales = 10.0
    cfg.filters.max_peg_ratio = 2.5

    dates = pd.bdate_range("2025-01-01", periods=260)
    rows = []
    for ticker in ["PASS_ALIAS", "FAIL_ALIAS"]:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "adj_close": 100.0,
                    "volume": 500_000,
                }
            )
    prices = pd.DataFrame(rows)

    funds = pd.DataFrame(
        [
            {
                "ticker": "PASS_ALIAS",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPe": 28.0,
                "priceToSales": 6.5,
                "trailingPegRatio": 1.9,
            },
            {
                "ticker": "FAIL_ALIAS",
                "marketCap": 10_000_000_000,
                "netIncomeToCommon": 100_000_000,
                "trailingPe": 85.0,
                "priceToSales": 13.0,
                "trailingPegRatio": 3.4,
            },
        ]
    )

    keep = apply_hard_filters(df_fund=funds, df_price=prices, cfg=cfg)
    assert keep.tolist() == ["PASS_ALIAS"]
