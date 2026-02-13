from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from borsa.backtest import run_monthly_backtest
from borsa.config import AppConfig
from borsa.data_fetch import get_or_fetch_benchmark, get_or_fetch_fundamentals, get_or_fetch_prices
from borsa.factors import build_factor_frame, filter_diagnostics
from borsa.scoring import FACTOR_SPECS, build_factor_scores, compute_final_scores, select_top_n
from borsa.universe import get_nasdaq100_tickers, normalize_for_yahoo

st.set_page_config(page_title="Borsa Value Screener", layout="wide")


@st.cache_data(show_spinner=False)
def _load_rank_inputs(
    as_of: date,
    use_cache: bool,
    period: str,
    custom_tickers: tuple[str, ...] | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    cfg = AppConfig()
    cfg.ensure_dirs()
    if custom_tickers:
        tickers = sorted(set(custom_tickers))
    else:
        tickers = get_nasdaq100_tickers(as_of=as_of)
    prices, _ = get_or_fetch_prices(tickers=tickers, as_of=as_of, cache_dir=cfg.cache_dir, period=period, use_cache=use_cache)
    funds, _ = get_or_fetch_fundamentals(tickers=tickers, as_of=as_of, cache_dir=cfg.cache_dir, use_cache=use_cache)
    selected = set(tickers)
    if not prices.empty and "ticker" in prices.columns:
        prices = prices[prices["ticker"].isin(selected)].copy()
    if not funds.empty and "ticker" in funds.columns:
        funds = funds[funds["ticker"].isin(selected)].copy()
    return tickers, prices, funds


@st.cache_data(show_spinner=False)
def _load_benchmark(as_of: date, use_cache: bool, period: str = "10y") -> pd.DataFrame:
    cfg = AppConfig()
    cfg.ensure_dirs()
    bench, _ = get_or_fetch_benchmark(ticker="QQQ", as_of=as_of, cache_dir=cfg.cache_dir, period=period, use_cache=use_cache)
    return bench


@st.cache_data(show_spinner=False)
def _load_snapshot_inputs(
    as_of: date,
    use_cache: bool,
    period: str,
    focus_ticker: str,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    cfg = AppConfig()
    cfg.ensure_dirs()
    ndx = get_nasdaq100_tickers(as_of=as_of)
    ticker = normalize_for_yahoo(focus_ticker)
    tickers = sorted(set([*ndx, ticker]))
    prices, _ = get_or_fetch_prices(tickers=tickers, as_of=as_of, cache_dir=cfg.cache_dir, period=period, use_cache=use_cache)
    funds, _ = get_or_fetch_fundamentals(tickers=tickers, as_of=as_of, cache_dir=cfg.cache_dir, use_cache=use_cache)
    return tickers, prices, funds


def _apply_sidebar_config() -> tuple[AppConfig, dict[str, object]]:
    cfg = AppConfig()
    cfg.ensure_dirs()

    st.sidebar.header("Run Settings")
    as_of = st.sidebar.date_input("As-of date", value=date.today())
    top_n = st.sidebar.slider("Top N", min_value=1, max_value=30, value=10, step=1)
    use_cache = st.sidebar.checkbox("Use cache", value=True)
    universe_mode = st.sidebar.radio("Universe", options=["Nasdaq-100", "Custom tickers"], index=0)
    custom_ticker_text = st.sidebar.text_area(
        "Custom tickers (comma/newline separated)",
        value="AAPL, MSFT, GOOGL",
        help="Used only when Universe = Custom tickers",
    )

    st.sidebar.header("Value Filters")
    cfg.filters.max_trailing_pe = st.sidebar.number_input("Max trailing P/E", min_value=5.0, max_value=80.0, value=float(cfg.filters.max_trailing_pe), step=1.0)
    cfg.filters.max_price_to_sales = st.sidebar.number_input("Max P/S", min_value=1.0, max_value=30.0, value=float(cfg.filters.max_price_to_sales), step=0.5)
    cfg.filters.max_peg_ratio = st.sidebar.number_input("Max PEG", min_value=0.5, max_value=8.0, value=float(cfg.filters.max_peg_ratio), step=0.1)

    st.sidebar.header("Backtest")
    bt_start = st.sidebar.date_input("Backtest start", value=date(2021, 1, 1))
    bt_end = st.sidebar.date_input("Backtest end", value=date(2026, 1, 31))
    cost_bps = st.sidebar.number_input("Cost (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

    return cfg, {
        "as_of": as_of,
        "top_n": top_n,
        "use_cache": use_cache,
        "universe_mode": universe_mode,
        "custom_ticker_text": custom_ticker_text,
        "bt_start": bt_start,
        "bt_end": bt_end,
        "cost_bps": cost_bps,
    }


def _parse_custom_tickers(raw: str) -> tuple[str, ...]:
    if not raw:
        return tuple()
    parts = []
    for chunk in raw.replace("\n", ",").split(","):
        t = chunk.strip()
        if t:
            parts.append(normalize_for_yahoo(t))
    deduped = sorted(set(parts))
    return tuple(deduped)


def _factor_contribution_frame(topn: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in topn.iterrows():
        for name, weight, _ in FACTOR_SPECS:
            if name in r.index:
                rows.append(
                    {
                        "ticker": r["ticker"],
                        "factor": name,
                        "contribution": float(r[name]) * (weight / 100.0),
                    }
                )
    return pd.DataFrame(rows)


def _merge_raw_valuation(topn: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    left = topn.loc[:, ~topn.columns.duplicated(keep="first")].copy()
    right = fundamentals.loc[:, ~fundamentals.columns.duplicated(keep="first")].copy()

    value_cols: list[str] = [c for c in ["ticker", "trailingPE", "priceToSalesTrailing12Months"] if c in right.columns]
    raw = right[value_cols].copy() if value_cols else right[["ticker"]].copy()

    peg_candidates = [c for c in ["pegRatio", "trailingPegRatio"] if c in right.columns]
    if peg_candidates:
        raw["raw_pegRatio"] = right[peg_candidates].bfill(axis=1).iloc[:, 0]
    if "trailingPE" in raw.columns:
        raw = raw.rename(columns={"trailingPE": "raw_trailingPE"})
    if "priceToSalesTrailing12Months" in raw.columns:
        raw = raw.rename(columns={"priceToSalesTrailing12Months": "raw_priceToSalesTrailing12Months"})

    merged = left.merge(raw, on="ticker", how="left")
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    return merged


def _top_table_column_config(columns: list[str]) -> dict[str, object]:
    config: dict[str, object] = {}
    if "rank" in columns:
        config["rank"] = st.column_config.NumberColumn(
            "Rank",
            help="Final position after applying all filters and weighted scoring (1 is best).",
            format="%d",
        )
    if "ticker" in columns:
        config["ticker"] = st.column_config.TextColumn(
            "Ticker",
            help="Stock symbol in Yahoo-compatible format.",
        )
    if "final_score" in columns:
        config["final_score"] = st.column_config.NumberColumn(
            "Final Score",
            help="Composite model score after weighted factors and missing-data penalty.",
            format="%.2f",
        )
    if "value_score" in columns:
        config["value_score"] = st.column_config.NumberColumn(
            "Value Score",
            help="Weighted sub-score (0-100) for valuation factors: P/E, P/S, PEG, and FCF yield.",
            format="%.1f",
        )
    if "quality_score" in columns:
        config["quality_score"] = st.column_config.NumberColumn(
            "Quality Score",
            help="Weighted sub-score (0-100) for profitability/quality: ROE, ROA, and margins.",
            format="%.1f",
        )
    if "growth_score" in columns:
        config["growth_score"] = st.column_config.NumberColumn(
            "Growth Score",
            help="Weighted sub-score (0-100) for revenue and earnings growth.",
            format="%.1f",
        )
    if "stability_score" in columns:
        config["stability_score"] = st.column_config.NumberColumn(
            "Stability Score",
            help="Weighted sub-score (0-100) for balance-sheet and volatility risk.",
            format="%.1f",
        )
    if "momentum_score" in columns:
        config["momentum_score"] = st.column_config.NumberColumn(
            "Momentum Score",
            help="Weighted sub-score (0-100) for 12-1 month momentum.",
            format="%.1f",
        )
    if "raw_trailingPE" in columns:
        config["raw_trailingPE"] = st.column_config.NumberColumn(
            "P/E (Raw)",
            help="Raw trailing Price-to-Earnings ratio from Yahoo fundamentals.",
            format="%.2f",
        )
    if "raw_priceToSalesTrailing12Months" in columns:
        config["raw_priceToSalesTrailing12Months"] = st.column_config.NumberColumn(
            "P/S (Raw)",
            help="Raw trailing Price-to-Sales ratio (TTM) from Yahoo fundamentals.",
            format="%.2f",
        )
    if "raw_pegRatio" in columns:
        config["raw_pegRatio"] = st.column_config.NumberColumn(
            "PEG (Raw)",
            help="Raw PEG ratio from Yahoo fundamentals.",
            format="%.2f",
        )
    if "returnOnEquity" in columns:
        config["returnOnEquity"] = st.column_config.NumberColumn(
            "ROE Score",
            help="Percentile score (0-100) for Return on Equity, normalized to handle raw percent-vs-decimal source inconsistencies.",
            format="%.1f",
        )
    if "returnOnAssets" in columns:
        config["returnOnAssets"] = st.column_config.NumberColumn(
            "ROA Score",
            help="Percentile score (0-100) for Return on Assets; higher is better.",
            format="%.1f",
        )
    if "operatingMargins" in columns:
        config["operatingMargins"] = st.column_config.NumberColumn(
            "Operating Margin Score",
            help="Percentile score (0-100) for operating margins; higher is better.",
            format="%.1f",
        )
    if "grossMargins" in columns:
        config["grossMargins"] = st.column_config.NumberColumn(
            "Gross Margin Score",
            help="Percentile score (0-100) for gross margins; higher is better.",
            format="%.1f",
        )
    if "revenueGrowth" in columns:
        config["revenueGrowth"] = st.column_config.NumberColumn(
            "Revenue Growth Score",
            help="Percentile score (0-100) for revenue growth; higher is better.",
            format="%.1f",
        )
    if "earningsGrowth" in columns:
        config["earningsGrowth"] = st.column_config.NumberColumn(
            "Earnings Growth Score",
            help="Percentile score (0-100) for earnings growth; higher is better.",
            format="%.1f",
        )
    return config


def _run_rank(cfg: AppConfig, args: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    custom_tickers: tuple[str, ...] | None = None
    if args["universe_mode"] == "Custom tickers":
        custom_tickers = _parse_custom_tickers(str(args["custom_ticker_text"]))
    tickers, prices, funds = _load_rank_inputs(
        as_of=args["as_of"],
        use_cache=args["use_cache"],
        period=cfg.price_period,
        custom_tickers=custom_tickers,
    )
    diag = filter_diagnostics(df_fund=funds, df_price=prices, cfg=cfg)
    factors = build_factor_frame(df_price=prices, df_fund=funds, cfg=cfg)
    scored = build_factor_scores(factors=factors, cfg=cfg)
    ranked = compute_final_scores(scores=scored)
    topn = select_top_n(final_df=ranked, n=min(int(args["top_n"]), max(1, len(tickers))))
    topn["rebalance_date"] = args["as_of"].isoformat()
    topn = _merge_raw_valuation(topn=topn, fundamentals=funds)
    return topn, ranked, diag, prices, len(tickers)


def _run_backtest(cfg: AppConfig, args: dict[str, object], prices: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    bench = _load_benchmark(as_of=args["as_of"], use_cache=args["use_cache"], period="10y")
    result = run_monthly_backtest(
        prices=prices,
        fundamentals=funds,
        benchmark_prices=bench,
        cfg=cfg,
        start=args["bt_start"],
        end=args["bt_end"],
        top_n=int(args["top_n"]),
        cost_bps=float(args["cost_bps"]),
    )
    perf = result.performance.copy()
    if perf.empty:
        return perf
    perf["rebalance_date"] = pd.to_datetime(perf["rebalance_date"])
    perf["strategy_equity"] = (1.0 + perf["strategy_return"].fillna(0.0)).cumprod()
    perf["benchmark_equity"] = (1.0 + perf["benchmark_return"].fillna(0.0)).cumprod()
    return perf


def _build_risk_gain_frame(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty or "ticker" not in ranked.columns:
        return pd.DataFrame()

    df = ranked.copy()

    gain_components = [c for c in ["value_score", "quality_score", "growth_score", "momentum_score"] if c in df.columns]
    if gain_components:
        df["gain_score"] = df[gain_components].mean(axis=1)
    else:
        fallback_gain = [c for c in ["final_score", "score", "mom_12_1", "revenueGrowth", "earningsGrowth"] if c in df.columns]
        if not fallback_gain:
            return pd.DataFrame()
        df["gain_score"] = df[fallback_gain].mean(axis=1)

    if "stability_score" in df.columns:
        df["risk_score"] = (100.0 - df["stability_score"]).clip(lower=0.0, upper=100.0)
    else:
        fallback_risk = [c for c in ["beta", "vol_63d", "max_drawdown_252d", "debtToEquity"] if c in df.columns]
        if fallback_risk:
            # In this pipeline those columns are already "higher is better" percentile scores.
            # Convert to risk by inversion.
            df["risk_score"] = (100.0 - df[fallback_risk].mean(axis=1)).clip(lower=0.0, upper=100.0)
        else:
            df["risk_score"] = 50.0

    df["reward_to_risk"] = df["gain_score"] / (df["risk_score"] + 1.0)
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.1, 25, 50, 75, 100],
        labels=["Low risk", "Medium risk", "High risk", "Very high risk"],
    )
    return df


def _ticker_monthly_snapshot_dates(
    prices: pd.DataFrame,
    ticker: str,
    as_of: date,
    months: int = 12,
) -> list[pd.Timestamp]:
    if prices.empty:
        return []
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px[(px["ticker"] == ticker) & (px["date"] <= pd.Timestamp(as_of))]
    if px.empty:
        return []
    month_ends = px.groupby(px["date"].dt.to_period("M"))["date"].max().sort_values()
    snapshot_dates: list[pd.Timestamp] = []
    asof_period = pd.Timestamp(as_of).to_period("M")
    for m in range(months, 0, -1):
        target = asof_period - m
        if target in month_ends.index:
            snapshot_dates.append(month_ends.loc[target])
    return snapshot_dates


def _price_column(df_prices: pd.DataFrame) -> str:
    if "adj_close" in df_prices.columns:
        return "adj_close"
    return "close"


def _build_snapshot_price_map(prices: pd.DataFrame, ticker: str) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    px_col = _price_column(prices)
    px = prices[prices["ticker"] == ticker][["date", px_col]].copy()
    if px.empty:
        return pd.Series(dtype=float)
    px["date"] = pd.to_datetime(px["date"])
    px = px.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return pd.to_numeric(px[px_col], errors="coerce")


def _compute_ticker_snapshot_history(
    cfg: AppConfig,
    as_of: date,
    ticker: str,
    use_cache: bool,
    months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker = normalize_for_yahoo(ticker)
    _, prices, funds = _load_snapshot_inputs(as_of=as_of, use_cache=use_cache, period="10y", focus_ticker=ticker)
    dates = _ticker_monthly_snapshot_dates(prices=prices, ticker=ticker, as_of=as_of, months=months)
    if not dates:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    price_map = _build_snapshot_price_map(prices, ticker)
    current_price = float(price_map.iloc[-1]) if not price_map.empty else float("nan")

    for snap_date in dates:
        hist = prices[pd.to_datetime(prices["date"]) <= snap_date].copy()
        factors = build_factor_frame(df_price=hist, df_fund=funds, cfg=cfg)
        scored = build_factor_scores(factors=factors, cfg=cfg)
        ranked = compute_final_scores(scores=scored)
        rg = _build_risk_gain_frame(ranked)

        row = ranked[ranked["ticker"] == ticker].head(1)
        row_rg = rg[rg["ticker"] == ticker].head(1)
        close_px = float(price_map.loc[snap_date]) if snap_date in price_map.index else float("nan")

        rec: dict = {
            "snapshot_date": snap_date.date().isoformat(),
            "ticker": ticker,
            "included": not row.empty,
            "close_price": close_px,
            "return_to_asof": (current_price / close_px - 1.0) if close_px and close_px == close_px and current_price == current_price else float("nan"),
        }
        if not row.empty:
            for c in [
                "rank",
                "final_score",
                "value_score",
                "quality_score",
                "growth_score",
                "stability_score",
                "momentum_score",
                "trailingPE",
                "priceToSalesTrailing12Months",
                "pegRatio",
            ]:
                if c in row.columns:
                    rec[c] = row.iloc[0][c]
        if not row_rg.empty:
            for c in ["gain_score", "risk_score", "reward_to_risk", "risk_band"]:
                if c in row_rg.columns:
                    rec[c] = row_rg.iloc[0][c]
        rows.append(rec)

    snap_df = pd.DataFrame(rows).sort_values("snapshot_date").reset_index(drop=True)
    expected_cols = [
        "rank",
        "final_score",
        "value_score",
        "quality_score",
        "growth_score",
        "stability_score",
        "momentum_score",
        "gain_score",
        "risk_score",
        "reward_to_risk",
        "risk_band",
        "trailingPE",
        "priceToSalesTrailing12Months",
        "pegRatio",
    ]
    for col in expected_cols:
        if col not in snap_df.columns:
            snap_df[col] = float("nan")

    if not snap_df.empty:
        snap_df["snapshot_date"] = pd.to_datetime(snap_df["snapshot_date"])
        snap_df["next_month_return"] = snap_df["close_price"].shift(-1) / snap_df["close_price"] - 1.0
        snap_df["snapshot_date"] = snap_df["snapshot_date"].dt.date.astype(str)

    return snap_df, funds


def _render_ticker_snapshot_screen(cfg: AppConfig, args: dict[str, object]) -> None:
    st.subheader("Ticker 12-Month Snapshot Check")
    st.caption("Run monthly historical snapshots to inspect how the model would have scored a ticker over the last year.")

    c1, c2 = st.columns([2, 1])
    input_ticker = c1.text_input("Ticker", value="AAPL", help="Any Yahoo-compatible ticker symbol.")
    months = c2.slider("Months back", min_value=6, max_value=24, value=12, step=1)

    run = st.button("Run 12M Snapshot", type="primary")
    if not run:
        st.info("Enter a ticker and click Run 12M Snapshot.")
        return

    ticker = normalize_for_yahoo(input_ticker.strip())
    if not ticker:
        st.error("Please enter a valid ticker.")
        return

    with st.spinner(f"Calculating snapshot history for {ticker}..."):
        try:
            hist_df, funds = _compute_ticker_snapshot_history(
                cfg=cfg,
                as_of=args["as_of"],
                ticker=ticker,
                use_cache=bool(args["use_cache"]),
                months=months,
            )
        except Exception as exc:
            st.error(f"Unable to compute snapshots right now: {exc}")
            st.info("Tip: enable cache or check network access, then retry.")
            return

    if hist_df.empty:
        st.warning("No snapshot data available for this ticker in the selected period.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Snapshots", f"{len(hist_df)}")
    included_ratio = 100.0 * hist_df["included"].mean()
    m2.metric("Included in Model", f"{included_ratio:.0f}%")
    if "next_month_return" in hist_df.columns and hist_df["next_month_return"].notna().any():
        m3.metric("Avg Next-Month Return", f"{hist_df['next_month_return'].mean() * 100:.2f}%")
    else:
        m3.metric("Avg Next-Month Return", "N/A")

    if "final_score" in hist_df.columns and hist_df["final_score"].notna().any():
        fig_score = px.line(
            hist_df,
            x="snapshot_date",
            y="final_score",
            markers=True,
            title=f"{ticker} Final Score over time",
        )
        fig_score.update_layout(height=320, xaxis_title="Snapshot date", yaxis_title="Final score")
        st.plotly_chart(fig_score, width="stretch")
    else:
        st.info("This ticker did not pass model filters in the selected snapshots, so no Final Score trend is available.")

    if "rank" in hist_df.columns:
        rank_df = hist_df.copy()
        fig_rank = px.line(rank_df, x="snapshot_date", y="rank", markers=True, title=f"{ticker} Rank over time")
        fig_rank.update_layout(height=320, xaxis_title="Snapshot date", yaxis_title="Rank (lower is better)")
        fig_rank.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_rank, width="stretch")

    if {"risk_score", "gain_score"}.issubset(hist_df.columns):
        fig_rg = px.scatter(
            hist_df,
            x="risk_score",
            y="gain_score",
            color="snapshot_date",
            size="reward_to_risk" if "reward_to_risk" in hist_df.columns else None,
            title=f"{ticker} Risk vs Gain by snapshot",
            hover_data={"snapshot_date": True, "rank": True, "final_score": ":.2f", "next_month_return": ":.2%"},
        )
        fig_rg.update_layout(height=360, xaxis_title="Risk score (higher riskier)", yaxis_title="Gain score")
        st.plotly_chart(fig_rg, width="stretch")

    peg_col = "pegRatio" if "pegRatio" in funds.columns else "trailingPegRatio" if "trailingPegRatio" in funds.columns else None
    raw_cols = [c for c in ["ticker", "trailingPE", "priceToSalesTrailing12Months"] if c in funds.columns]
    if peg_col:
        raw_cols.append(peg_col)
    raw_row = funds[funds["ticker"] == ticker]
    if not raw_row.empty and raw_cols:
        raw_display = raw_row[raw_cols].copy()
        if peg_col and peg_col != "pegRatio":
            raw_display = raw_display.rename(columns={peg_col: "pegRatio"})
        st.caption("Current raw valuation fields from fundamentals snapshot")
        st.dataframe(raw_display, hide_index=True, width="stretch")

    st.caption("Snapshot table")
    show_cols = [
        c
        for c in [
            "snapshot_date",
            "included",
            "rank",
            "final_score",
            "value_score",
            "quality_score",
            "growth_score",
            "stability_score",
            "momentum_score",
            "gain_score",
            "risk_score",
            "reward_to_risk",
            "next_month_return",
            "return_to_asof",
        ]
        if c in hist_df.columns
    ]
    st.dataframe(hist_df[show_cols], hide_index=True, width="stretch")

    with st.expander("How this snapshot check works"):
        st.markdown(
            """
- For each month in the selected lookback window, the app takes a month-end snapshot date.
- It recomputes the full cross-sectional ranking model using data available up to that date.
- It then extracts the selected ticker's score/rank for that snapshot.
- `next_month_return` is the realized price return to the next snapshot month.
- `return_to_asof` is the realized return from the snapshot date to the current as-of date.
- Fundamental fields are taken from the latest fundamentals snapshot (not true historical point-in-time fundamentals).
- This helps evaluate whether higher historical scores tended to align with better realized outcomes.
"""
        )


def _render_calculation_explanation(cfg: AppConfig, args: dict[str, object]) -> None:
    st.subheader("What This App Is Doing")
    universe_label = str(args["universe_mode"])
    selected_tickers = _parse_custom_tickers(str(args["custom_ticker_text"])) if universe_label == "Custom tickers" else tuple()
    settings_rows = [
        {"Setting": "As-of date", "Value": str(args["as_of"])},
        {"Setting": "Universe mode", "Value": universe_label},
        {"Setting": "Top N", "Value": str(args["top_n"])},
        {"Setting": "Use cache", "Value": "Yes" if bool(args["use_cache"]) else "No"},
        {"Setting": "Max P/E", "Value": f"{cfg.filters.max_trailing_pe:.2f}"},
        {"Setting": "Max P/S", "Value": f"{cfg.filters.max_price_to_sales:.2f}"},
        {"Setting": "Max PEG", "Value": f"{cfg.filters.max_peg_ratio:.2f}"},
        {"Setting": "Backtest start", "Value": str(args["bt_start"])},
        {"Setting": "Backtest end", "Value": str(args["bt_end"])},
        {"Setting": "Transaction cost (bps)", "Value": f"{float(args['cost_bps']):.1f}"},
    ]
    if selected_tickers:
        settings_rows.append({"Setting": "Custom tickers", "Value": ", ".join(selected_tickers)})

    st.caption("Current run settings")
    st.dataframe(pd.DataFrame(settings_rows), hide_index=True, width="stretch")

    st.subheader("Step-by-Step Calculation")
    st.markdown(
        """
1. **Universe selection**
   - Use either the full Nasdaq-100 universe or your custom ticker list.
   - Download prices and fundamentals from Yahoo Finance (or load from cache).

2. **Hard filters**
   - Remove names that fail liquidity/size/profitability rules.
   - Apply valuation caps (P/E, P/S, PEG) from the sidebar.

3. **Factor construction**
   - Build factors such as value (P/E, P/S, PEG, FCF yield), quality (ROE/ROA/margins), growth, momentum, and risk.
   - Normalize ratio fields (for example ROE) to handle source unit inconsistencies.

4. **Scoring**
   - Winsorize each factor (5th/95th percentile).
   - Convert factors to cross-sectional percentile scores (0-100).
   - Invert “lower is better” factors (e.g., P/E, volatility, debt).
   - Fill missing factors with neutral score (50), then apply missing-data penalty.
   - Compute final weighted composite score and rank descending.

5. **Displayed score groups**
   - **Value Score**: valuation-focused sub-score.
   - **Quality Score**: profitability/efficiency sub-score.
   - **Growth Score**: revenue/earnings growth sub-score.
   - **Stability Score**: leverage/volatility risk sub-score.
   - **Momentum Score**: trend sub-score.

6. **Backtest (when run)**
   - Monthly rebalance to top-N ranked names.
   - Equal-weight portfolio, optional transaction-cost deduction (bps).
   - Compare strategy equity curve and drawdown versus QQQ.
"""
    )
    st.subheader("What Each Score Means")
    st.markdown(
        """
- **Final Score**: Overall ranking score after applying weights and missing-data penalties.
- **Value Score**: How attractive valuation looks (lower multiples and stronger cash-flow yield score higher).
- **Quality Score**: Profitability and efficiency (ROE, ROA, margins).
- **Growth Score**: Revenue and earnings growth strength.
- **Stability Score**: Balance-sheet and price-risk stability (debt, beta, volatility, drawdown).
- **Momentum Score**: 12-1 month trend signal.
"""
    )
    st.subheader("How to Read the Risk/Gain Dashboard")
    st.markdown(
        """
- **Gain Score**: A blended upside proxy from Value, Quality, Growth, and Momentum sub-scores.
- **Risk Score**: A downside proxy derived primarily from stability (debt, volatility, drawdown, beta).
- **Reward/Risk**: `Gain Score / (Risk Score + 1)`. Higher usually means better upside per unit of model risk.
- **Scatter chart**:
  - Move **up** for stronger upside profile.
  - Move **left** for lower estimated risk.
  - Bigger bubble means stronger reward/risk.
- This is a model-based heuristic to compare candidates, not a guarantee of future returns.
"""
    )
    with st.expander("Important caveats"):
        st.markdown(
            """
- Data is sourced from Yahoo and may contain delays, revisions, or field inconsistencies.
- This is a ranking model, not a complete investment process.
- Backtest results use current fundamentals snapshot and can differ from true point-in-time historical values.
- Use this as a research aid, then validate each candidate with your own due diligence.
"""
        )


def main() -> None:
    st.title("Borsa: Long-Term Value Screener")
    st.caption("Nasdaq-100 ranking and backtest dashboard")

    cfg, args = _apply_sidebar_config()
    tab_dashboard, tab_ticker, tab_info = st.tabs(["Dashboard", "Ticker 12M Check", "How It Works"])

    with tab_dashboard:
        left, right = st.columns([1, 1])
        run_rank = left.button("Run Rank", type="primary", width="stretch")
        run_backtest = right.button("Run Backtest", width="stretch")

        if run_rank:
            can_run = True
            if args["universe_mode"] == "Custom tickers":
                chosen = _parse_custom_tickers(str(args["custom_ticker_text"]))
                if not chosen:
                    st.error("Please enter at least one valid ticker in Custom tickers mode.")
                    can_run = False
            if can_run:
                with st.spinner("Running ranking model..."):
                    topn, ranked, diag, prices, universe_size = _run_rank(cfg, args)
                    st.session_state["_topn"] = topn
                    st.session_state["_ranked"] = ranked
                    st.session_state["_diag"] = diag
                    st.session_state["_prices"] = prices
                    st.session_state["_universe_size"] = universe_size

        if run_backtest:
            can_run = True
            if args["universe_mode"] == "Custom tickers":
                chosen = _parse_custom_tickers(str(args["custom_ticker_text"]))
                if not chosen:
                    st.error("Please enter at least one valid ticker in Custom tickers mode.")
                    can_run = False
            if can_run:
                with st.spinner("Loading rank inputs first..."):
                    topn, ranked, diag, prices, universe_size = _run_rank(cfg, args)
                    st.session_state["_topn"] = topn
                    st.session_state["_ranked"] = ranked
                    st.session_state["_diag"] = diag
                    st.session_state["_prices"] = prices
                    st.session_state["_universe_size"] = universe_size

                custom_tickers: tuple[str, ...] | None = None
                if args["universe_mode"] == "Custom tickers":
                    custom_tickers = _parse_custom_tickers(str(args["custom_ticker_text"]))
                funds = _load_rank_inputs(
                    as_of=args["as_of"],
                    use_cache=args["use_cache"],
                    period=cfg.price_period,
                    custom_tickers=custom_tickers,
                )[2]
                with st.spinner("Running backtest..."):
                    perf = _run_backtest(cfg, args, st.session_state["_prices"], funds)
                    st.session_state["_perf"] = perf

        if "_topn" not in st.session_state:
            st.info("Set filters in the sidebar and click Run Rank.")
        else:
            topn = st.session_state["_topn"]
            ranked = st.session_state["_ranked"]
            diag = st.session_state["_diag"]
            universe_size = st.session_state["_universe_size"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Universe", f"{universe_size}")
            c2.metric("After Filters", f"{len(ranked)}")
            c3.metric("Top N", f"{len(topn)}")

            st.subheader("Top Candidates")
            show_cols = [
                c
                for c in [
                    "rank",
                    "ticker",
                    "final_score",
                    "value_score",
                    "quality_score",
                    "growth_score",
                    "stability_score",
                    "momentum_score",
                    "raw_trailingPE",
                    "raw_priceToSalesTrailing12Months",
                    "raw_pegRatio",
                    "returnOnEquity",
                    "returnOnAssets",
                    "operatingMargins",
                    "grossMargins",
                    "revenueGrowth",
                    "earningsGrowth",
                ]
                if c in topn.columns
            ]
            display_df = topn[show_cols].sort_values("rank")
            st.dataframe(
                display_df,
                width="stretch",
                column_config=_top_table_column_config(show_cols),
                hide_index=True,
            )

            st.subheader("Filter Funnel")
            fig_funnel = px.bar(diag, x="step", y="count", text="count")
            fig_funnel.update_layout(height=320, xaxis_title="", yaxis_title="Tickers")
            st.plotly_chart(fig_funnel, width="stretch")

            st.subheader("Factor Contribution (Top N)")
            contrib = _factor_contribution_frame(topn)
            fig_contrib = px.bar(contrib, x="ticker", y="contribution", color="factor", title="Weighted score contribution by factor")
            fig_contrib.update_layout(height=420)
            st.plotly_chart(fig_contrib, width="stretch")

            if "_perf" in st.session_state and not st.session_state["_perf"].empty:
                perf = st.session_state["_perf"]
                st.subheader("Backtest Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=perf["rebalance_date"], y=perf["strategy_equity"], mode="lines", name="Strategy"))
                fig_eq.add_trace(go.Scatter(x=perf["rebalance_date"], y=perf["benchmark_equity"], mode="lines", name="QQQ"))
                fig_eq.update_layout(height=420, yaxis_title="Growth of $1")
                st.plotly_chart(fig_eq, width="stretch")

                st.subheader("Drawdown")
                perf = perf.copy()
                perf["strategy_dd"] = perf["strategy_equity"] / perf["strategy_equity"].cummax() - 1.0
                perf["benchmark_dd"] = perf["benchmark_equity"] / perf["benchmark_equity"].cummax() - 1.0
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=perf["rebalance_date"], y=perf["strategy_dd"], mode="lines", name="Strategy DD"))
                fig_dd.add_trace(go.Scatter(x=perf["rebalance_date"], y=perf["benchmark_dd"], mode="lines", name="QQQ DD"))
                fig_dd.update_layout(height=360, yaxis_title="Drawdown")
                st.plotly_chart(fig_dd, width="stretch")

            st.subheader("Risk/Gain Dashboard")
            risk_gain = _build_risk_gain_frame(ranked)
            if risk_gain.empty:
                st.info("Not enough data to build risk/gain analysis for this run.")
            else:
                top_rg = risk_gain.sort_values("rank").head(len(topn))
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Gain Score (Top N)", f"{top_rg['gain_score'].mean():.1f}")
                m2.metric("Avg Risk Score (Top N)", f"{top_rg['risk_score'].mean():.1f}")
                m3.metric("Avg Reward/Risk (Top N)", f"{top_rg['reward_to_risk'].mean():.2f}")

                fig_rg = px.scatter(
                    risk_gain,
                    x="risk_score",
                    y="gain_score",
                    size="reward_to_risk",
                    color="final_score",
                    hover_name="ticker",
                    hover_data={
                        "rank": True,
                        "final_score": ":.2f",
                        "risk_score": ":.1f",
                        "gain_score": ":.1f",
                        "reward_to_risk": ":.2f",
                        "risk_band": True,
                    },
                    title="Risk vs Gain Map (all ranked tickers)",
                )
                fig_rg.update_layout(height=460, xaxis_title="Risk Score (higher = riskier)", yaxis_title="Gain Score (higher = stronger upside profile)")
                st.plotly_chart(fig_rg, width="stretch")

                st.caption("Reward/Risk is a model-derived proxy: Gain Score divided by Risk Score + 1.")
                with st.expander("How to interpret this Risk/Gain section"):
                    st.markdown(
                        """
- **Gain Score** combines valuation, quality, growth, and momentum sub-scores.
- **Risk Score** reflects model-estimated downside risk (higher means riskier).
- **Reward/Risk** compares upside profile against risk profile.
- A practical preference is often names in the **upper-left** (higher gain, lower risk),
  but final decisions should include your own thesis, horizon, and risk tolerance.
"""
                    )
                rr_cols = [c for c in ["ticker", "rank", "final_score", "gain_score", "risk_score", "reward_to_risk", "risk_band"] if c in risk_gain.columns]
                st.dataframe(
                    risk_gain[rr_cols].sort_values("reward_to_risk", ascending=False).head(15),
                    hide_index=True,
                    width="stretch",
                )

    with tab_info:
        _render_calculation_explanation(cfg, args)

    with tab_ticker:
        _render_ticker_snapshot_screen(cfg, args)


if __name__ == "__main__":
    main()
