from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


FIGURE_FILES = {
    "horizon": "figure_1_horizon_comparison.png",
    "features": "figure_2_feature_ic.png",
    "macro": "figure_3_macro_ic.png",
    "lstm": "figure_4_lstm_diagnostics.png",
    "portfolio": "figure_5_portfolio_periods.png",
    "factor": "figure_6_factor_ic.png",
    "ensemble": "figure_7_final_ensemble.png",
    "candidates": "figure_8_candidate_comparison.png",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _no_data(ax: plt.Axes, message: str = "계산 가능한 결과 없음") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "Date" in result:
        result["Date"] = pd.to_datetime(result.Date, errors="coerce")
    return result


def _legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def _plot_horizon(tables: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = tables.get("profit_horizon_metrics", pd.DataFrame())
    summary = tables.get("profit_horizon_summary", pd.DataFrame())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if len(metrics) and {"horizon", "scope", "daily_ic"}.issubset(metrics):
        means = metrics.groupby(["horizon", "scope"], as_index=False).daily_ic.mean()
        for name in ["all_train", "stable_train"]:
            group = means.loc[means.scope.eq(name)].sort_values("horizon")
            axes[0].plot(group.horizon, group.daily_ic, "o-", label=name)
        axes[0].axhline(0, color="gray", ls="--")
        axes[0].set_xticks([20, 60, 120])
        axes[0].set_xlabel("타깃=보유=리밸런싱 거래일")
        axes[0].set_ylabel("평균 스냅샷 IC")
        _legend(axes[0])
    else:
        _no_data(axes[0])
    axes[0].set_title("기간별 평균 스냅샷 IC")
    if len(summary) and {"horizon", "strategy", "top_n", "mdd"}.issubset(summary):
        selected = summary.loc[summary.top_n.eq(10)]
        for name in ["all_train", "stable_train"]:
            group = selected.loc[selected.strategy.eq(name)].sort_values("horizon")
            axes[1].plot(group.horizon, group.mdd, "o-", label=name)
        axes[1].set_xticks([20, 60, 120])
        axes[1].set_xlabel("거래일")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        _legend(axes[1])
    else:
        _no_data(axes[1])
    axes[1].set_title("완전한 경로의 전체 MDD")
    _finish(fig, path)


def _plot_ic_cases(frame: pd.DataFrame, cases: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    frame = _date(frame)
    if len(frame) and {"Date", "case", "daily_ic"}.issubset(frame):
        for case in cases:
            group = frame.loc[frame.case.eq(case)].sort_values("Date")
            if len(group):
                ax.plot(group.Date, group.daily_ic, "o-", label=case)
        ax.axhline(0, color="gray", ls="--")
        _legend(ax)
        fig.autofmt_xdate()
    else:
        _no_data(ax)
    ax.set_title(title)
    ax.set_ylabel("OOS 스냅샷 IC")
    _finish(fig, path)


def _plot_lstm(tables: dict[str, pd.DataFrame], path: Path) -> None:
    losses = _date(tables.get("profit_lstm_losses", pd.DataFrame()))
    metrics = _date(tables.get("profit_lstm_metrics", pd.DataFrame()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if len(losses) and {"Date", "epoch", "validation_loss"}.issubset(losses):
        for day, group in losses.groupby("Date"):
            axes[0].plot(group.epoch, group.validation_loss, "o-", label=str(day.date()))
        _legend(axes[0])
        axes[0].set_xlabel("epoch")
    else:
        _no_data(axes[0])
    axes[0].set_title("LSTM 내부 검증 SmoothL1")
    if len(metrics) and {"Date", "case", "daily_ic"}.issubset(metrics):
        for case, group in metrics.groupby("case"):
            axes[1].plot(group.Date, group.daily_ic, "o-", label=case)
        axes[1].axhline(0, color="gray", ls="--")
        _legend(axes[1])
        fig.autofmt_xdate()
    else:
        _no_data(axes[1])
    axes[1].set_title("외부 IC: LSTM vs 동일표본 LGBM")
    _finish(fig, path)


def _plot_portfolio(tables: dict[str, pd.DataFrame], path: Path) -> None:
    periods = _date(tables.get("profit_horizon_periods", pd.DataFrame()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, horizon in zip(axes, [20, 60, 120]):
        if len(periods) and {"horizon", "strategy", "top_n", "assumption_loss50"}.issubset(periods):
            for name in ["all_train", "stable_train", "lowvol", "stable_lowvol"]:
                group = periods.loc[
                    periods.horizon.eq(horizon) & periods.top_n.eq(10) & periods.strategy.eq(name)
                ].sort_values("Date")
                if not len(group):
                    continue
                line, = ax.plot(group.Date, group.assumption_loss50, "o-", label=name)
                missing = group.return_settled.isna() if "return_settled" in group else pd.Series(False, index=group.index)
                ax.scatter(
                    group.loc[missing, "Date"], group.loc[missing, "assumption_loss50"],
                    marker="X", s=70, color=line.get_color(), edgecolor="black", zorder=5,
                )
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            if horizon == 20:
                _legend(ax)
        else:
            _no_data(ax)
        ax.set_title(f"{horizon}일 기간수익 / 미정산 -50%")
    fig.autofmt_xdate()
    _finish(fig, path)


def _plot_factor(tables: dict[str, pd.DataFrame], path: Path) -> None:
    frame = _date(tables.get("profit_factor_robustness", pd.DataFrame()))
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(frame) and {"Date", "raw_ic", "incremental_ic"}.issubset(frame):
        ax.plot(frame.Date, frame.raw_ic, "o-", label="원 IC")
        ax.plot(frame.Date, frame.incremental_ic, "o-", label="모멘텀·베타 통제 IC")
        ax.axhline(0, color="gray", ls="--")
        _legend(ax)
        fig.autofmt_xdate()
    else:
        _no_data(ax)
    ax.set_title("H5-1: 사후 부분 순위 IC (섹터 미통제)")
    _finish(fig, path)


def _plot_ensemble(tables: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = _date(tables.get("profit_final_ensemble_metrics", pd.DataFrame()))
    periods = _date(tables.get("profit_final_ensemble_periods", pd.DataFrame()))
    recommendation = tables.get("profit_final_ensemble_recommendation", pd.DataFrame()).copy()
    exposure = tables.get("profit_final_ensemble_sector_exposure", pd.DataFrame()).copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    if len(metrics) and {"Date", "daily_ic"}.issubset(metrics):
        axes[0, 0].plot(metrics.Date, metrics.daily_ic, "o-", color="tab:blue")
        axes[0, 0].axhline(0, color="gray", ls="--")
    else:
        _no_data(axes[0, 0])
    axes[0, 0].set_title("선택 모델 OOS Fold IC")
    if len(periods) and {"Date", "return_settled", "assumption_loss50"}.issubset(periods):
        axes[0, 1].plot(periods.Date, periods.assumption_loss50, "o-", label="미정산 -50% 민감도")
        settled = periods.return_settled.notna()
        axes[0, 1].scatter(periods.loc[settled, "Date"], periods.loc[settled, "return_settled"], label="정산 수익", zorder=5)
        axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
        _legend(axes[0, 1])
    else:
        _no_data(axes[0, 1])
    axes[0, 1].set_title("선택 모델 OOS 기간 수익")
    if len(recommendation) and {"Ticker", "pred"}.issubset(recommendation):
        shown = recommendation.sort_values("pred").tail(10)
        axes[1, 0].barh(shown.Ticker, shown.pred, color="tab:green")
        axes[1, 0].set_xlim(0, 1)
    else:
        _no_data(axes[1, 0])
    axes[1, 0].set_title("최신 Stable 선택 모델 Top-10")
    if len(exposure) and {"Sector", "weight"}.issubset(exposure):
        exposure = exposure.sort_values("weight", ascending=False)
        axes[1, 1].bar(exposure.Sector, exposure.weight, color="tab:orange")
        axes[1, 1].axhline(0.5, color="red", ls="--", label="Sector 50% 상한")
        axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, 1].tick_params(axis="x", rotation=35)
        _legend(axes[1, 1])
    else:
        _no_data(axes[1, 1])
    axes[1, 1].set_title("최신 추천 Sector 노출")
    fig.autofmt_xdate()
    _finish(fig, path)


def save_figures(tables: dict[str, pd.DataFrame], figure_dir: Path, *, refresh: bool = False) -> dict[str, str]:
    """Save the six source-notebook plots plus the final ensemble dashboard."""
    _setup_style()
    figure_dir.mkdir(parents=True, exist_ok=refresh)
    _plot_horizon(tables, figure_dir / FIGURE_FILES["horizon"])
    _plot_ic_cases(
        tables.get("profit_controlled_models", pd.DataFrame()),
        ["base_excess_lgbm", "difference_features", "volume_features"],
        "H2: 피처별 외부 스냅샷 IC",
        figure_dir / FIGURE_FILES["features"],
    )
    _plot_ic_cases(
        tables.get("profit_macro_metrics", pd.DataFrame()),
        ["micro_base", "macro_added", "interactions_added"],
        "거시/상호작용 추가: 동일표본 외부 IC",
        figure_dir / FIGURE_FILES["macro"],
    )
    _plot_lstm(tables, figure_dir / FIGURE_FILES["lstm"])
    _plot_portfolio(tables, figure_dir / FIGURE_FILES["portfolio"])
    _plot_factor(tables, figure_dir / FIGURE_FILES["factor"])
    _plot_ensemble(tables, figure_dir / FIGURE_FILES["ensemble"])
    summary = tables.get('profit_candidate_summary', pd.DataFrame())
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, column, title in zip(axes, ['cumulative_return', 'mdd', 'mean_ic'],
                                  ['전체 누적수익 (미정산 -50% 가정)', '전체 MDD (가정 경로 포함)', '평균 OOS IC (관측 라벨)']):
        if len(summary) and column in summary:
            ax.barh(summary.strategy, summary[column])
            ax.axvline(0, color='gray', linewidth=.7)
            if column != 'mean_ic':
                ax.xaxis.set_major_formatter(PercentFormatter(1))
        else:
            _no_data(ax)
        ax.set_title(title)
    _finish(fig, figure_dir / FIGURE_FILES['candidates'])
    return {key: f"figures/{name}" for key, name in FIGURE_FILES.items()}
