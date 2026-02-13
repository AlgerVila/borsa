# borsa

`borsa` is a Python project that builds a Nasdaq-100 stock ranking pipeline from Yahoo Finance data.

It currently supports:
- Universe ingestion (Nasdaq-100 constituents)
- Yahoo price + fundamentals data ingestion with local parquet cache
- Factor engineering and composite scoring
- Top-N ranking output (CSV + JSON)
- Monthly walk-forward backtest vs `QQQ`

## Investment profile in this version

The current model is configured for a **multi-year value hold** style:
- Strong emphasis on valuation and quality
- Lower emphasis on short-term momentum
- Strict valuation hard filters before ranking

This is a research tool, not financial advice.

## Project layout

- `src/borsa/config.py`: Global config, filter defaults, lookbacks, paths
- `src/borsa/universe.py`: Nasdaq-100 universe fetch + ticker normalization
- `src/borsa/data_fetch.py`: Yahoo ingestion and cache helpers
- `src/borsa/factors.py`: Hard filters + factor construction
- `src/borsa/scoring.py`: Winsorization, percentile ranking, weighted score
- `src/borsa/backtest.py`: Monthly walk-forward backtest engine
- `src/borsa/reporting.py`: Output writers
- `src/borsa/cli.py`: CLI commands (`rank`, `backtest`)
- `tests/`: Unit/integration-style tests for scoring/filtering/backtest behavior

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run tests:

```bash
pytest -q
```

## Data and cache

Cache directory:
- `/Users/algerino/repo/borsa/data/cache`

Common cache files:
- `prices_2y_YYYYMMDD.parquet`
- `prices_10y_YYYYMMDD.parquet`
- `fundamentals_YYYYMMDD.parquet`
- `benchmark_QQQ_10y_YYYYMMDD.parquet`

## Current factor model

### Hard filters (applied first)

From `HardFilters` defaults in `src/borsa/config.py`:
- `marketCap >= 5B`
- `avg daily dollar volume (20d) >= 20M`
- Positive net income (`netIncomeToCommon > 0`) when present
- `0 < trailingPE <= 30`
- `0 < priceToSalesTrailing12Months <= 10`
- `0 < PEG <= 2.5`
  - PEG is normalized from Yahoo aliases (`pegRatio` / `trailingPegRatio`)
- Missing-factor ratio cap: `<= 40%`

### Scoring factors and weights (long-term value profile)

From `FACTOR_SPECS` in `src/borsa/scoring.py`:
- Value:
  - `trailingPE` (20, lower is better)
  - `priceToSalesTrailing12Months` (15, lower is better)
  - `pegRatio` (10, lower is better)
- Quality:
  - `returnOnEquity` (15, higher is better)
  - `operatingMargins` (10, higher is better)
  - `revenueGrowth` (10, higher is better)
- Momentum / risk control:
  - `mom_12_1` (10, higher is better)
  - `beta` (4, lower is better)
  - `vol_63d` (4, lower is better)
  - `max_drawdown_252d` (2, lower is better)

### Normalization and score construction

- Winsorize each factor at 5th/95th percentile
- Convert to cross-sectional percentile ranks
- For “lower is better” factors, invert rank (`100 - rank`)
- Fill missing factor score with `50`
- Missing penalty: `-2` per missing factor, capped at `-10`
- Final score = weighted factor score + missing penalty

## CLI usage

### 1) Rank top candidates

```bash
borsa rank --as-of 2026-02-13 --top 10 --use-cache
```

Outputs:
- `/Users/algerino/repo/borsa/output/top10_20260213.csv`
- `/Users/algerino/repo/borsa/output/top10_20260213.json`

### 2) Backtest strategy vs QQQ

```bash
borsa backtest --start 2021-01-01 --end 2026-01-31 --top 10 --cost-bps 10 --as-of 2026-02-13 --use-cache
```

Outputs:
- `/Users/algerino/repo/borsa/output/backtest_performance_20260213.csv`
- `/Users/algerino/repo/borsa/output/backtest_metrics_20260213.csv`

## Frontend dashboard

The project now includes a Streamlit dashboard:
- `/Users/algerino/repo/borsa/app.py`

Run it:

```bash
source .venv/bin/activate
streamlit run /Users/algerino/repo/borsa/app.py
```

Dashboard capabilities:
- Sidebar controls for:
  - `as_of` date, top-N, cache usage
  - universe mode: `Nasdaq-100` or `Custom tickers`
  - custom tickers input (comma/newline separated, 1..N symbols)
  - value caps (`P/E`, `P/S`, `PEG`)
  - backtest date range and transaction cost
- Rank workflow:
  - top candidates table with raw valuation fields
  - filter funnel chart (survivors after each rule)
  - factor contribution stacked bar chart
- Backtest workflow:
  - strategy vs QQQ equity curve
  - strategy vs QQQ drawdown chart

## Backtest method (current)

- Rebalance frequency: monthly (month-end trading date)
- Portfolio: equal-weight top `N` names (`N=10` default)
- Transaction costs: `cost_bps` applied as turnover cost each rebalance
- Benchmark: `QQQ`
- Reported metrics:
  - CAGR
  - Volatility
  - Sharpe
  - Max drawdown
  - Average turnover

## Notes on data quality and caveats

- Yahoo fields are not point-in-time perfect and may be revised.
- Some symbols can fail due to source outages or ticker changes.
- Universe source is Wikipedia with a local fallback snapshot.
- Column names from Yahoo can vary (handled for PEG alias).
- Current backtest uses latest fundamentals snapshot, not historical fundamentals per rebalance.

## Recent run summary (as of 2026-02-13)

- Tests: `6 passed`
- Rank run:
  - Universe size: `94`
  - Ranked after filters: `31`
  - Top-10 written to output files above
- Backtest run (`2021-01-01` to `2026-01-31`):
  - 60 monthly periods
  - Metrics written to `backtest_metrics_20260213.csv`

## Next improvements

- Add point-in-time fundamentals per rebalance date
- Add configurable model profiles (`balanced`, `value`, `quality`, `growth`)
- Add parameter sweep/optimization utility for filter caps and weights
- Add richer reporting (equity curve chart, factor attribution by period)
