from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import research_data as rd

from .config import ProjectConfig


@dataclass
class AnalysisData:
    frame: pd.DataFrame
    calendar: pd.DatetimeIndex
    market: pd.Series
    membership: pd.DataFrame
    sectors: pd.DataFrame
    macro: pd.DataFrame
    input_hashes: dict[str, str]


def _build_required_features(price_path: Path, benchmark_path: Path) -> pd.DataFrame:
    """Memory-bounded subset of research_data.build_features used by the suites.

    The formulas and input files match test_validation_report; columns unused by
    either merged hypothesis suite are not materialised.
    """
    raw = pd.read_parquet(price_path, columns=["Date", "Ticker", "Close", "Volume"])
    raw["Date"] = pd.to_datetime(raw.Date)
    market = pd.read_parquet(benchmark_path).sort_values("Date").set_index("Date").Close
    calendar = pd.DatetimeIndex(market.index)
    market_log_return = np.log(market / market.shift()).fillna(0.0)
    parts = []
    ticker_values = raw.Ticker.to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ticker_values[1:] != ticker_values[:-1]) + 1, len(raw)]
    if len(boundaries) - 1 != raw.Ticker.nunique():
        raise ValueError("final_df.parquet은 Ticker별 연속 블록으로 정렬되어야 합니다.")
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = raw.iloc[left:right]
        ticker = ticker_values[left]
        indexed = group.set_index("Date").reindex(calendar)
        close = indexed.Close
        log_return = np.log(close / close.shift())
        indexed["log_return"] = log_return.astype("float32")
        indexed["market_return"] = market_log_return.astype("float32")
        delta = close.diff()
        volume = indexed.Volume.astype(float).where(indexed.Volume.gt(0))
        previous_mean = volume.shift(1).rolling(20, min_periods=20).mean()
        indexed["relative_volume_20d"] = (volume / previous_mean - 1).astype("float32")
        indexed["volume_trend_20_120"] = (
            volume.rolling(20, min_periods=20).mean()
            / volume.rolling(120, min_periods=120).mean()
            - 1
        ).astype("float32")
        for window in [20, 30, 60, 120, 252]:
            indexed[f"vol_ann_{window}d"] = (
                log_return.rolling(window).std() * np.sqrt(252)
            ).astype("float32")
            indexed[f"beta_{window}d"] = (
                log_return.rolling(window).cov(market_log_return)
                / market_log_return.rolling(window).var().replace(0, np.nan)
            ).astype("float32")
            indexed[f"ma_gap_{window}d"] = (
                close / close.rolling(window).mean() - 1
            ).astype("float32")
            indexed[f"momentum_{window}d"] = (close / close.shift(window) - 1).astype("float32")
            indexed[f"relative_momentum_{window}d"] = (
                log_return.rolling(window).sum() - market_log_return.rolling(window).sum()
            ).astype("float32")
            if window == 120:
                indexed["downside_vol_120d"] = np.sqrt(
                    log_return.clip(upper=0).pow(2).rolling(window).mean() * 252
                ).astype("float32")
                indexed["annual_mean_120d"] = (
                    log_return.rolling(window).mean() * 252
                ).astype("float32")
                indexed["sortino_120d"] = (
                    indexed["annual_mean_120d"] / (indexed["downside_vol_120d"] + 1e-9)
                ).astype("float32")
        for window in [14, 28, 56]:
            up = delta.clip(lower=0).rolling(window).mean()
            down = (-delta.clip(upper=0)).rolling(window).mean()
            indexed[f"rsi_{window}d"] = (
                (100 - 100 / (1 + up / down))
                .where(down.ne(0), 100)
                .mask(up.eq(0) & down.eq(0), 50)
                .astype("float32")
            )
        for prefix in ["momentum", "relative_momentum"]:
            for label, short, long in [("short", 20, 60), ("mid", 60, 120), ("long", 120, 252)]:
                indexed[f"{prefix}_diff_{label}"] = (
                    indexed[f"{prefix}_{short}d"] - indexed[f"{prefix}_{long}d"]
                ).astype("float32")
        indexed = indexed.loc[group.Date].copy(deep=False)
        indexed["Ticker"] = ticker
        parts.append(indexed.rename_axis("Date").reset_index())
    result = pd.concat(parts, ignore_index=True)
    numeric = result.select_dtypes(include="number").columns
    for column in numeric:
        values = result[column].to_numpy(copy=False)
        invalid = ~np.isfinite(values)
        if invalid.any():
            result.loc[invalid, column] = np.nan
    return result.reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_sectors(path: Path) -> pd.DataFrame:
    sectors = pd.read_csv(path)
    rename = {}
    if "ticker" in sectors.columns:
        rename["ticker"] = "Ticker"
    if "sector" in sectors.columns:
        rename["sector"] = "Sector"
    sectors = sectors.rename(columns=rename)
    if not {"Ticker", "Sector"}.issubset(sectors.columns):
        raise ValueError("sp500_universe.csv에는 ticker/sector 컬럼이 필요합니다.")
    sectors = sectors[["Ticker", "Sector"]].copy()
    sectors["Ticker"] = (
        sectors.Ticker.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    sectors["Sector"] = sectors.Sector.fillna("UNKNOWN").astype(str)
    return sectors.drop_duplicates("Ticker", keep="last")


def load_analysis_data(config: ProjectConfig) -> AnalysisData:
    """Build the only analysis panel used by both formerly separate notebooks."""
    config.validate()
    frame = _build_required_features(config.raw_price_path, config.benchmark_path)
    membership = rd.load_membership(config.membership_path)
    frame = rd.mark_membership(frame, membership, "2026-06-30")
    frame["Ticker"] = (
        frame.Ticker.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    if frame.duplicated(["Date", "Ticker"]).any():
        raise ValueError("공통 패널에 Date/Ticker 중복이 있습니다.")

    market_frame = pd.read_parquet(config.benchmark_path).copy()
    market_frame["Date"] = pd.to_datetime(market_frame.Date)
    market = market_frame.sort_values("Date").set_index("Date").Close
    calendar = pd.DatetimeIndex(market.index.unique()).sort_values()
    sectors = _normalise_sectors(config.sector_path)
    macro = pd.read_csv(config.macro_path, index_col=0, parse_dates=True)

    hashes = {
        str(p.relative_to(config.project_root)): _sha256(p)
        for p in [
            config.raw_price_path,
            config.benchmark_path,
            config.membership_path,
            config.sector_path,
            config.macro_path,
        ]
    }
    return AnalysisData(frame, calendar, market, membership, sectors, macro, hashes)


def coverage_table(data: AnalysisData) -> pd.DataFrame:
    coverage = rd.coverage(data.frame, data.membership, "2026-06-30")
    coverage["coverage_ratio"] = coverage.price_available / coverage.expected
    return (
        coverage.groupby(coverage.Date.dt.year)
        .agg(
            mean_price_coverage=("coverage_ratio", "mean"),
            min_price_coverage=("coverage_ratio", "min"),
            mean_missing_securities=("missing", "mean"),
        )
        .reset_index(names="year")
    )


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.Series(
        np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1), index=frame.index
    )
