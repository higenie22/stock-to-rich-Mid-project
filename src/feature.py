"""Rolling profitability and stability features for daily OHLCV data.

The module creates four metrics over 60- and 120-trading-day windows:
cumulative return, Sharpe ratio, annualized volatility, and maximum drawdown.
Every value uses only observations available on or before that row's date.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (60, 120)
TRADING_DAYS_PER_YEAR = 252
REQUIRED_COLUMNS = {"Date", "Ticker", "Close"}


def feature_columns(windows: Iterable[int] = DEFAULT_WINDOWS) -> list[str]:
    """Return the feature column names produced for ``windows``."""
    return [
        f"{metric}_{window}d"
        for window in windows
        for metric in ("return", "sharpe", "volatility", "mdd")
    ]


def _rolling_mdd(prices: np.ndarray) -> float:
    """Calculate maximum drawdown inside one rolling price window."""
    running_peak = np.maximum.accumulate(prices)
    drawdowns = prices / running_peak - 1.0
    return float(drawdowns.min())


def add_features(
    prices: pd.DataFrame,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    annualization: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Add eight rolling return/risk features to a daily price panel.

    Parameters
    ----------
    prices:
        DataFrame containing at least ``Date``, ``Ticker``, and ``Close``.
    windows:
        Positive trading-day windows. The default (60, 120) creates 8 fields.
    annualization:
        Trading days used to annualize volatility and the Sharpe ratio.
    risk_free_rate:
        Annual risk-free rate as a decimal. It is converted to a daily rate.

    Notes
    -----
    A window of N means N closing-price observations. Return therefore compares
    the first and last close within those N observations. Features remain NaN
    until a ticker has a complete window. MDD is stored as a non-positive rate.
    """
    missing = sorted(REQUIRED_COLUMNS - set(prices.columns))
    if missing:
        raise ValueError(f"피처 생성 필수 컬럼 누락: {missing}")

    normalized_windows = tuple(int(window) for window in windows)
    if not normalized_windows or any(window < 2 for window in normalized_windows):
        raise ValueError("windows에는 2 이상의 기간이 하나 이상 필요합니다.")
    if len(set(normalized_windows)) != len(normalized_windows):
        raise ValueError("windows에 중복 기간이 있습니다.")
    if annualization <= 0:
        raise ValueError("annualization은 양수여야 합니다.")

    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if frame[["Date", "Ticker", "Close"]].isna().any().any():
        raise ValueError("Date, Ticker, Close에는 결측 또는 변환 불가 값이 없어야 합니다.")
    if (frame["Close"] <= 0).any():
        raise ValueError("Close는 모두 양수여야 합니다.")
    if frame.duplicated(["Ticker", "Date"]).any():
        raise ValueError("(Ticker, Date) 중복 행이 있습니다.")

    frame = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / annualization) - 1.0

    for window in normalized_windows:
        ticker_group = frame.groupby("Ticker", sort=False, group_keys=False)
        frame[f"return_{window}d"] = ticker_group["Close"].transform(
            lambda close: close / close.shift(window - 1) - 1.0
        )

        daily_returns = ticker_group["Close"].pct_change(fill_method=None)
        returns_group = daily_returns.groupby(frame["Ticker"], sort=False)
        mean_return = returns_group.transform(
            lambda values: values.rolling(window - 1, min_periods=window - 1).mean()
        )
        return_std = returns_group.transform(
            lambda values: values.rolling(window - 1, min_periods=window - 1).std(ddof=1)
        )
        frame[f"volatility_{window}d"] = return_std * np.sqrt(annualization)
        frame[f"sharpe_{window}d"] = (
            (mean_return - daily_risk_free) / return_std * np.sqrt(annualization)
        ).where(return_std > 0)
        frame[f"mdd_{window}d"] = ticker_group["Close"].transform(
            lambda close: close.rolling(window, min_periods=window).apply(_rolling_mdd, raw=True)
        )

    return frame


def build_feature_dataset(
    input_path: str = "data/processed/final_df.parquet",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Load processed prices, create the default features, and optionally save."""
    featured = add_features(pd.read_parquet(input_path))
    if output_path is not None:
        featured.to_parquet(output_path, index=False)
    return featured

