from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FactorWeights(BaseModel):
    momentum: float = 0.35
    quality: float = 0.25
    value: float = 0.20
    risk: float = 0.20


class HardFilters(BaseModel):
    min_market_cap: float = 5_000_000_000
    min_avg_daily_dollar_volume: float = 20_000_000
    require_positive_net_income: bool = True
    require_pe_bounds: bool = True
    min_trailing_pe: float = 0.0
    max_trailing_pe: float = 30.0
    require_ps_cap: bool = True
    max_price_to_sales: float = 10.0
    require_peg_cap: bool = True
    max_peg_ratio: float = 2.5
    max_missing_factor_ratio: float = 0.40


class Lookbacks(BaseModel):
    trading_days_year: int = 252
    momentum_short: int = 63
    momentum_mid: int = 126
    momentum_long: int = 252
    skip_recent_days: int = 21
    risk_vol_window: int = 63


class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path | None = None
    cache_dir: Path | None = None
    output_dir: Path | None = None

    weights: FactorWeights = Field(default_factory=FactorWeights)
    filters: HardFilters = Field(default_factory=HardFilters)
    lookbacks: Lookbacks = Field(default_factory=Lookbacks)

    fundamentals_period: str = "2y"
    price_period: str = "2y"

    def model_post_init(self, __context: object) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def cache_file_for(kind: str, as_of: date, cache_dir: Path) -> Path:
    stamp = as_of.strftime("%Y%m%d")
    return cache_dir / f"{kind}_{stamp}.parquet"
