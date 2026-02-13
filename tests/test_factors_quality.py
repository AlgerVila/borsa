import pandas as pd

from borsa.factors import compute_quality_value_factors


def test_quality_factors_normalize_ratios_and_aliases() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "returnOnEquity": 25.0,  # percent-point representation
                "returnOnAssets": 5.0,  # percent-point representation
                "operatingMargins": 30.0,  # percent-point representation
                "grossMargins": 0.40,  # decimal representation
                "revenueGrowth": 15.0,  # percent-point representation
                "earningsGrowth": 0.12,  # decimal representation
                "trailingPE": 20.0,
                "priceToSalesTrailing12Months": 4.0,
                "trailingPegRatio": 1.8,
                "debtToEquity": 140.0,  # percent representation
                "freeCashflow": 2_000_000_000,
                "marketCap": 20_000_000_000,
            }
        ]
    )
    out = compute_quality_value_factors(df).iloc[0]

    assert round(float(out["returnOnEquity"]), 4) == 0.25
    assert round(float(out["returnOnAssets"]), 4) == 0.05
    assert round(float(out["operatingMargins"]), 4) == 0.30
    assert round(float(out["grossMargins"]), 4) == 0.40
    assert round(float(out["revenueGrowth"]), 4) == 0.15
    assert round(float(out["earningsGrowth"]), 4) == 0.12
    assert round(float(out["pegRatio"]), 4) == 1.8
    assert round(float(out["debtToEquity"]), 4) == 1.4
    assert round(float(out["fcf_yield"]), 4) == 0.10


def test_quality_factors_use_pe_ps_alias_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trailingPe": 28.0,
                "priceToSales": 6.5,
                "trailingPegRatio": 1.7,
            }
        ]
    )
    out = compute_quality_value_factors(df).iloc[0]

    assert round(float(out["trailingPE"]), 4) == 28.0
    assert round(float(out["priceToSalesTrailing12Months"]), 4) == 6.5
    assert round(float(out["pegRatio"]), 4) == 1.7
