from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def output_paths(output_dir: Path, as_of: date) -> tuple[Path, Path]:
    stamp = as_of.strftime("%Y%m%d")
    csv_path = output_dir / f"top10_{stamp}.csv"
    json_path = output_dir / f"top10_{stamp}.json"
    return csv_path, json_path


def write_topn(topn: pd.DataFrame, output_dir: Path, as_of: date) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output_paths(output_dir=output_dir, as_of=as_of)
    topn.to_csv(csv_path, index=False)
    topn.to_json(json_path, orient="records", indent=2, date_format="iso")
    return csv_path, json_path


def write_backtest(
    performance: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    as_of: date,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = as_of.strftime("%Y%m%d")
    perf_path = output_dir / f"backtest_performance_{stamp}.csv"
    metrics_path = output_dir / f"backtest_metrics_{stamp}.csv"
    performance.to_csv(perf_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return perf_path, metrics_path
