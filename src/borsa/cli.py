from __future__ import annotations

from datetime import date, datetime

import typer

from borsa.backtest import run_monthly_backtest
from borsa.config import AppConfig
from borsa.data_fetch import get_or_fetch_benchmark, get_or_fetch_fundamentals, get_or_fetch_prices
from borsa.factors import build_factor_frame
from borsa.reporting import write_backtest, write_topn
from borsa.scoring import build_factor_scores, compute_final_scores, select_top_n
from borsa.universe import get_nasdaq100_tickers

app = typer.Typer(help="borsa CLI")


def _parse_date(value: str, field: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"`{field}` must be in YYYY-MM-DD format") from exc


@app.command()
def rank(
    as_of: str = typer.Option(date.today().isoformat(), help="Snapshot date (YYYY-MM-DD) for cache/output naming"),
    top: int = typer.Option(10, min=1, help="Number of tickers to keep in final output"),
    use_cache: bool = typer.Option(True, help="Use cached parquet files when available"),
) -> None:
    as_of_date = _parse_date(as_of, "as_of")
    cfg = AppConfig()
    cfg.ensure_dirs()

    tickers = get_nasdaq100_tickers(as_of=as_of_date)
    prices, prices_path = get_or_fetch_prices(
        tickers=tickers,
        as_of=as_of_date,
        cache_dir=cfg.cache_dir,
        period=cfg.price_period,
        use_cache=use_cache,
    )
    funds, funds_path = get_or_fetch_fundamentals(
        tickers=tickers,
        as_of=as_of_date,
        cache_dir=cfg.cache_dir,
        use_cache=use_cache,
    )

    factors = build_factor_frame(df_price=prices, df_fund=funds, cfg=cfg)
    factor_scores = build_factor_scores(factors=factors, cfg=cfg)
    ranked = compute_final_scores(scores=factor_scores)
    topn = select_top_n(final_df=ranked, n=top)
    topn["rebalance_date"] = as_of_date.isoformat()
    csv_path, json_path = write_topn(topn=topn, output_dir=cfg.output_dir, as_of=as_of_date)

    typer.echo(f"Universe size: {len(tickers)}")
    typer.echo(f"Prices rows: {len(prices)} (cache: {prices_path})")
    typer.echo(f"Fundamentals rows: {len(funds)} (cache: {funds_path})")
    typer.echo(f"Ranked rows: {len(ranked)}")
    typer.echo(f"Top {min(top, len(topn))} written to: {csv_path}")
    typer.echo(f"JSON output: {json_path}")


@app.command()
def backtest(
    start: str = typer.Option(..., help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Backtest end date (YYYY-MM-DD)"),
    as_of: str = typer.Option(date.today().isoformat(), help="Snapshot date (YYYY-MM-DD) for cache/output naming"),
    top: int = typer.Option(10, min=1, help="Number of tickers in the strategy portfolio"),
    cost_bps: float = typer.Option(10.0, min=0.0, help="Transaction cost in basis points per rebalance"),
    use_cache: bool = typer.Option(True, help="Use cached parquet files when available"),
) -> None:
    start_date = _parse_date(start, "start")
    end_date = _parse_date(end, "end")
    as_of_date = _parse_date(as_of, "as_of")
    if end_date <= start_date:
        raise typer.BadParameter("`end` must be after `start`")

    cfg = AppConfig()
    cfg.ensure_dirs()

    tickers = get_nasdaq100_tickers(as_of=as_of_date)
    prices, prices_path = get_or_fetch_prices(
        tickers=tickers,
        as_of=as_of_date,
        cache_dir=cfg.cache_dir,
        period="10y",
        use_cache=use_cache,
    )
    funds, funds_path = get_or_fetch_fundamentals(
        tickers=tickers,
        as_of=as_of_date,
        cache_dir=cfg.cache_dir,
        use_cache=use_cache,
    )
    bench, bench_path = get_or_fetch_benchmark(
        ticker="QQQ",
        as_of=as_of_date,
        cache_dir=cfg.cache_dir,
        period="10y",
        use_cache=use_cache,
    )

    result = run_monthly_backtest(
        prices=prices,
        fundamentals=funds,
        benchmark_prices=bench,
        cfg=cfg,
        start=start_date,
        end=end_date,
        top_n=top,
        cost_bps=cost_bps,
    )

    perf_path, metrics_path = write_backtest(
        performance=result.performance,
        metrics=result.metrics,
        output_dir=cfg.output_dir,
        as_of=as_of_date,
    )

    typer.echo(f"Universe size: {len(tickers)}")
    typer.echo(f"Prices rows: {len(prices)} (cache: {prices_path})")
    typer.echo(f"Fundamentals rows: {len(funds)} (cache: {funds_path})")
    typer.echo(f"Benchmark rows: {len(bench)} (cache: {bench_path})")
    typer.echo(f"Backtest periods: {len(result.performance)}")
    typer.echo(f"Performance output: {perf_path}")
    typer.echo(f"Metrics output: {metrics_path}")


if __name__ == "__main__":
    app()
