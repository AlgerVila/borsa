import pandas as pd

from borsa.config import AppConfig
from borsa.scoring import build_factor_scores, compute_final_scores, select_top_n


def test_missing_penalty_and_rank_order() -> None:
    cfg = AppConfig()
    factors = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "mom_12_1": [0.20, 0.10, 0.05],
            "mom_6_1": [0.15, 0.03, 0.01],
            "mom_3_1": [0.08, 0.02, 0.0],
            "returnOnEquity": [0.25, 0.10, 0.05],
            "operatingMargins": [0.30, 0.15, 0.10],
            "revenueGrowth": [0.20, 0.05, 0.03],
            "trailingPE": [20.0, 35.0, 50.0],
            "priceToSalesTrailing12Months": [4.0, 8.0, 12.0],
            "pegRatio": [1.2, 2.0, 3.0],
            "beta": [1.0, 1.2, 1.5],
            "vol_63d": [0.20, 0.30, 0.40],
            "max_drawdown_252d": [0.15, 0.25, 0.35],
        }
    )

    factors.loc[2, "revenueGrowth"] = None
    scores = build_factor_scores(factors, cfg)
    ranked = compute_final_scores(scores)

    assert set(ranked["ticker"]) == {"AAA", "BBB", "CCC"}
    assert ranked.iloc[0]["ticker"] == "AAA"
    ccc = ranked.loc[ranked["ticker"] == "CCC"].iloc[0]
    assert ccc["missing_penalty"] == -2.0
    for col in ["value_score", "quality_score", "growth_score", "stability_score", "momentum_score"]:
        assert col in ranked.columns


def test_select_top_n() -> None:
    frame = pd.DataFrame({"ticker": ["A", "B", "C"], "final_score": [90, 80, 70]})
    out = select_top_n(frame, n=2)
    assert out["ticker"].tolist() == ["A", "B"]
