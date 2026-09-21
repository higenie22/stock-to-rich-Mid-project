from __future__ import annotations

import gc
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .config import ProjectConfig
from .data_pipeline import load_analysis_data
from .stability import run_stability_suite


COLORS = {
    "primary": "#2457A7",
    "exploratory": "#E68A2E",
    "neutral": "#A8B0BA",
    "good": "#2A9D6F",
    "bad": "#C94C4C",
}


def _setup_style() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic" if platform.system() == "Windows" else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    # Presentation export: increase every default text class by exactly 2 pt.
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FAFBFC"


def _read_table(run_dir: Path, name: str, supplied: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if name in supplied and isinstance(supplied[name], pd.DataFrame):
        return supplied[name].copy()
    path = run_dir / "tables" / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ensemble_figure(run_dir: Path, supplied: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table = _read_table(run_dir, "cv_locked_oos_comparison", supplied)
    required = {"strategy", "cumulative_return", "mdd"}
    if table.empty or not required.issubset(table.columns):
        raise ValueError("cv_locked_oos_comparison 표가 없어 앙상블 비교를 그릴 수 없습니다.")
    table = table.dropna(subset=["cumulative_return", "mdd"]).copy()
    table["evaluation_role"] = np.where(
        table.strategy.eq("cv_locked_relu_4model"),
        "CV에서 비중 고정 후 OOS 평가",
        np.where(table.strategy.str.startswith("dynamic4"), "OOS 성과를 사용한 탐색 비교", "고정/단독 비교"),
    )
    labels = {
        "base_lgbm": "Base LightGBM",
        "elasticnet": "ElasticNet",
        "elasticnet_lgbm_equal": "ElasticNet+LGBM 동일비중",
        "elasticnet_lgbm_lstm_equal": "ElasticNet+LGBM+LSTM 동일비중",
        "original_40_40_20": "기존 40/40/20",
        "dynamic4_relu": "동적 4모델 ReLU (탐색)",
        "dynamic4_softmax": "동적 4모델 Softmax (탐색)",
        "cv_locked_relu_4model": "CV 고정 4모델 (최종 후보)",
    }
    table["label"] = table.strategy.map(labels).fillna(table.strategy)
    table = table.sort_values("cumulative_return")
    colors = [
        COLORS["primary"] if s == "cv_locked_relu_4model" else
        COLORS["exploratory"] if str(s).startswith("dynamic4") else COLORS["neutral"]
        for s in table.strategy
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    axes[0].barh(table.label, table.cumulative_return, color=colors)
    axes[0].axvline(0, color="#555", linewidth=.8)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("동일 4개 OOS 기간 누적수익률")
    axes[0].set_xlabel("누적수익률")
    for y, value in enumerate(table.cumulative_return):
        axes[0].text(value, y, f" {value:.1%}", va="center", fontsize=11)

    axes[1].barh(table.label, table.mdd.abs(), color=colors)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("최대낙폭 크기 — 낮을수록 우수")
    axes[1].set_xlabel("|MDD|")
    for y, value in enumerate(table.mdd.abs()):
        axes[1].text(value, y, f" {value:.1%}", va="center", fontsize=11)

    for row, color in zip(table.itertuples(), colors):
        axes[2].scatter(abs(row.mdd), row.cumulative_return, s=90, color=color, edgecolor="white", linewidth=.8)
        axes[2].annotate(row.label, (abs(row.mdd), row.cumulative_return), xytext=(5, 5), textcoords="offset points", fontsize=10)
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_xlabel("|MDD| — 왼쪽이 우수")
    axes[2].set_ylabel("누적수익률 — 위쪽이 우수")
    axes[2].set_title("위험–수익 위치")
    for ax in axes:
        ax.grid(alpha=.18, axis="x")
    fig.suptitle("앙상블 모델 성과 비교: 확정용과 탐색용을 구분", fontsize=18, fontweight="bold")
    fig.text(.5, .01, "파란색: CV에서 가중치를 고정한 최종 후보 · 주황색: OOS 성과를 비중 결정에 사용한 탐색 결과", ha="center", fontsize=12)
    fig.tight_layout(rect=[0, .04, 1, .94])
    _save(fig, run_dir / "figures" / "presentation_ensemble_comparison.png")
    return table


def _cv_locked_final_figure(run_dir: Path, supplied: dict[str, pd.DataFrame]) -> None:
    """Visualize the actual CV-locked final candidate, not the OOS-selected dynamic model."""
    metrics = _read_table(run_dir, "cv_locked_oos_metrics", supplied)
    periods = _read_table(run_dir, "cv_locked_oos_periods", supplied)
    weights = _read_table(run_dir, "cv_locked_weights", supplied)
    comparison = _read_table(run_dir, "cv_locked_oos_comparison", supplied)
    required = [metrics, periods, weights, comparison]
    if any(table.empty for table in required):
        raise ValueError("CV 고정 최종 모델의 metrics/periods/weights/comparison 표가 필요합니다.")

    metrics["Date"] = pd.to_datetime(metrics.Date)
    periods["Date"] = pd.to_datetime(periods.Date)
    periods = periods.sort_values("Date")
    final_compare = comparison.loc[
        comparison.strategy.isin(["elasticnet", "cv_locked_relu_4model"])
    ].dropna(subset=["cumulative_return", "mdd"]).copy()
    model_labels = {
        "elasticnet": "ElasticNet 단독",
        "cv_locked_relu_4model": "CV 고정 4모델",
    }

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    axes[0, 0].plot(metrics.Date, metrics.daily_ic, "o-", color=COLORS["primary"], linewidth=2.2)
    axes[0, 0].axhline(0, color="#666", linestyle="--", linewidth=1)
    axes[0, 0].set_title("최종 모델 OOS Fold IC")
    axes[0, 0].set_ylabel("Spearman IC")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for row in metrics.itertuples():
        axes[0, 0].annotate(f"{row.daily_ic:+.3f}", (row.Date, row.daily_ic),
                            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=11)

    bar_colors = [COLORS["bad"] if int(x) > 0 else COLORS["primary"] for x in periods.assumed_holdings]
    bars = axes[0, 1].bar(periods.Date.dt.strftime("%Y-%m-%d"), periods.selection_return, color=bar_colors)
    axes[0, 1].axhline(0, color="#666", linewidth=1)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].set_title("최종 모델 OOS 기간 수익")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for bar, value, assumed in zip(bars, periods.selection_return, periods.assumed_holdings):
        note = " (-50% 가정)" if int(assumed) else ""
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value,
                        f"{value:+.1%}{note}", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=11)

    weights = weights.copy()
    weight_labels = {"elasticnet": "ElasticNet", "mlp": "MLP", "macro": "Macro-LGBM", "lstm": "LSTM"}
    weights["label"] = weights.model.map(weight_labels).fillna(weights.model)
    bars = axes[1, 0].bar(weights.label, weights.selected_relu_weight, color=COLORS["primary"])
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].set_title("2020~2022 내부 CV로 고정한 앙상블 비중")
    axes[1, 0].set_ylabel("고정 비중")
    for bar, value, ic in zip(bars, weights.selected_relu_weight, weights.mean_cv_ic):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value,
                        f"{value:.1%}\nCV IC {ic:+.3f}", ha="center", va="bottom", fontsize=11)

    final_compare["label"] = final_compare.strategy.map(model_labels)
    x = np.arange(len(final_compare))
    width = .36
    b1 = axes[1, 1].bar(x - width / 2, final_compare.cumulative_return, width,
                        label="누적수익률", color=COLORS["primary"])
    b2 = axes[1, 1].bar(x + width / 2, final_compare.mdd.abs(), width,
                        label="|MDD|", color=COLORS["exploratory"])
    axes[1, 1].set_xticks(x, final_compare.label)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].set_title("같은 OOS 기간: ElasticNet 대비 최종 모델")
    axes[1, 1].legend(frameon=False)
    for bars_ in [b1, b2]:
        for bar in bars_:
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=11)

    for ax in axes.flat:
        ax.grid(alpha=.18, axis="y")
    fig.suptitle("CV 고정 4모델 최종 후보 성과", fontsize=19, fontweight="bold")
    fig.text(.5, .01,
             "가중치는 내부 CV에서 확정 후 OOS에서 변경하지 않음 · 4개 120거래일 OOS · 미정산 1종목은 -50% 시나리오",
             ha="center", fontsize=12, color="#444")
    fig.tight_layout(rect=[0, .04, 1, .95])
    _save(fig, run_dir / "figures" / "presentation_cv_locked_final_model.png")


def _stability_figure(run_dir: Path, project_root: Path) -> dict[str, pd.DataFrame]:
    config = ProjectConfig(project_root=project_root, output_root=project_root / "outputs")
    data = load_analysis_data(config)
    try:
        tables, policy = run_stability_suite(data.frame, data.calendar, data.sectors, config)
        latest = tables["stability_handoff"].copy()
        latest["Date"] = pd.to_datetime(latest.Date)
        latest = latest.loc[latest.Date.eq(latest.Date.max())].copy()
    finally:
        del data
        gc.collect()

    vol_col = policy.volatility_column
    beta_col = policy.beta_column
    latest["abs_beta"] = latest[beta_col].abs()
    axis_corr = latest[[vol_col, "abs_beta"]].corr(method="spearman").iloc[0, 1]
    ablation = tables["stability_axis_ablation"].groupby("policy", as_index=False).agg(
        false_stable_rate=("false_stable_rate", "mean"),
        mean_future_mdd=("mean_future_mdd", "mean"),
        dates=("Date", "nunique"),
    )
    lookback = tables["stability_lookback"].copy()
    k_summary = tables["stability_k_sensitivity"].groupby("k", as_index=False).agg(
        silhouette=("silhouette", "mean"), dbi=("dbi", "mean"),
        min_cluster_share=("min_cluster_share", "mean"), dates=("Date", "nunique"),
    )
    groups = tables["stability_future_groups"].copy()
    sector = tables["stability_handoff"].groupby("Sector", as_index=False).agg(
        observations=("Ticker", "size"), stable_n=("stable", "sum")
    )
    sector["stable_share"] = sector.stable_n / sector.observations
    sector = sector.loc[~sector.Sector.eq("Unknown")].sort_values("stable_share")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    group_colors = {"Stable": COLORS["good"], "Middle": COLORS["neutral"], "Risky": COLORS["bad"]}
    for group, part in latest.groupby("stability_group"):
        axes[0, 0].scatter(part[vol_col], part.abs_beta, s=18, alpha=.65,
                           color=group_colors.get(group, COLORS["neutral"]), label=group)
    axes[0, 0].set_xlabel("120일 연환산 변동성")
    axes[0, 0].set_ylabel("|120일 베타|")
    axes[0, 0].set_title(f"H1 두 위험 축 (Spearman ρ={axis_corr:.2f})")
    axes[0, 0].legend(frameon=False)

    ablation_labels = {"volatility_only": "변동성 단독", "beta_only": "베타 단독", "canonical_2d_kmeans": "변동성+베타"}
    aa = ablation.assign(label=ablation.policy.map(ablation_labels)).sort_values("false_stable_rate")
    axes[0, 1].bar(aa.label, aa.false_stable_rate, color=[COLORS["good"], COLORS["neutral"], COLORS["primary"]])
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].set_title("H1 축 제거 실험: 미래 Risky 오분류율")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for x, value in enumerate(aa.false_stable_rate):
        axes[0, 1].text(x, value, f"{value:.1%}", ha="center", va="bottom")

    axes[0, 2].plot(lookback.lookback, lookback.false_stable_rate, "o-", color=COLORS["primary"], linewidth=2)
    axes[0, 2].scatter([120], lookback.loc[lookback.lookback.eq(120), "false_stable_rate"], s=130,
                       color=COLORS["exploratory"], zorder=3, label="채택 120일")
    axes[0, 2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 2].set_xticks(lookback.lookback)
    axes[0, 2].set_title("H2 Lookback별 미래 Risky 오분류율")
    axes[0, 2].set_xlabel("Lookback (거래일)")
    axes[0, 2].legend(frameon=False)

    axes[1, 0].plot(k_summary.k, k_summary.silhouette, "o-", color=COLORS["primary"], linewidth=2)
    if 3 in set(k_summary.k):
        axes[1, 0].scatter([3], k_summary.loc[k_summary.k.eq(3), "silhouette"], s=130,
                           color=COLORS["exploratory"], zorder=3, label="채택 K=3")
    axes[1, 0].set_xticks(k_summary.k)
    axes[1, 0].set_title("H3 K별 평균 Silhouette — 높을수록 우수")
    axes[1, 0].set_xlabel("군집 수 K")
    axes[1, 0].legend(frameon=False)

    gg = groups.set_index("stability_group").reindex(["Stable", "Middle", "Risky"]).reset_index()
    x = np.arange(len(gg))
    width = .36
    axes[1, 1].bar(x - width / 2, gg.mean_future_vol, width, label="미래 변동성", color=COLORS["primary"])
    axes[1, 1].bar(x + width / 2, gg.mean_future_mdd.abs(), width, label="미래 |MDD|", color=COLORS["bad"])
    axes[1, 1].set_xticks(x, gg.stability_group)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].set_title("H4 군집별 120일 미래 위험")
    axes[1, 1].legend(frameon=False)

    axes[1, 2].barh(sector.Sector, sector.stable_share, color=COLORS["primary"])
    axes[1, 2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 2].set_title("H5 섹터별 Stable 분류 비중")
    axes[1, 2].set_xlabel("전체 평가시점 관측 중 Stable 비중")
    for ax in axes.flat:
        ax.grid(alpha=.18, axis="y")
    fig.suptitle("안정성 가설 검증 시각화", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .96])
    _save(fig, run_dir / "figures" / "presentation_stability_hypotheses.png")

    result = {**tables, "presentation_stability_axis_summary": ablation,
              "presentation_stability_k_summary": k_summary,
              "presentation_stability_sector_share": sector,
              "presentation_stability_latest_snapshot": latest}
    result["presentation_stability_axis_correlation"] = pd.DataFrame([{
        "Date": latest.Date.max(), "spearman_vol_abs_beta": axis_corr, "observations": len(latest)
    }])
    return result


def _research_figure(run_dir: Path, supplied: dict[str, pd.DataFrame], alpha_summary: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    internal = _read_table(run_dir, "internal_cv_ic", supplied)
    macro_cv = _read_table(run_dir, "macro_cv", supplied)
    controls = _read_table(run_dir, "controls_summary", supplied)
    factor = _read_table(run_dir, "robust_factor", supplied)
    if alpha_summary is None or alpha_summary.empty:
        alpha_summary = _read_table(run_dir, "alpha_summary", supplied)

    feature_rows = []
    if not internal.empty:
        for case, part in internal.loc[internal.case.isin(["base", "difference", "volume"])].groupby("case"):
            feature_rows.append({"experiment": "기술 피처", "case": case, "mean_cv_ic": part.ic.mean(), "observations": part.ic.notna().sum()})
    if not macro_cv.empty:
        for case, part in macro_cv.groupby("case"):
            feature_rows.append({"experiment": "거시 피처", "case": case, "mean_cv_ic": part.daily_ic.mean(), "observations": part.daily_ic.notna().sum()})
    feature = pd.DataFrame(feature_rows)
    settled_models = controls.loc[
        controls.strategy.isin(["base", "rolling", "ranker", "elasticnet"])
        & controls.cumulative_return.notna() & controls.mdd.notna()
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    if not feature.empty:
        labels = feature.experiment + ": " + feature.case
        colors = [COLORS["primary"] if c not in ["base"] else COLORS["neutral"] for c in feature.case]
        axes[0, 0].barh(labels, feature.mean_cv_ic, color=colors)
        axes[0, 0].axvline(0, color="#555", linewidth=.8)
        axes[0, 0].set_title("H7 추가 피처의 내부 CV 평균 IC")
        axes[0, 0].set_xlabel("평균 IC (실험군별 표본 정의 주의)")

    if not settled_models.empty:
        names = {"base": "Base LGBM", "rolling": "최근 3년 LGBM", "ranker": "LGBM Ranker", "elasticnet": "ElasticNet"}
        sm = settled_models.assign(label=settled_models.strategy.map(names)).sort_values("cumulative_return")
        axes[0, 1].barh(sm.label, sm.cumulative_return, color=[COLORS["neutral"] if x == "base" else COLORS["primary"] for x in sm.strategy])
        axes[0, 1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[0, 1].axvline(0, color="#555", linewidth=.8)
        axes[0, 1].set_title("H8·H9 학습 방식/구조: 완전 정산 OOS 누적수익")
        axes[0, 1].set_xlabel("누적수익률")

    if not factor.empty:
        factor = factor.copy()
        factor["Date"] = pd.to_datetime(factor.Date)
        axes[1, 0].plot(factor.Date, factor.raw_ic, "o-", label="원 IC", color=COLORS["primary"])
        axes[1, 0].plot(factor.Date, factor.incremental_ic, "o-", label="모멘텀·베타 통제 후 IC", color=COLORS["exploratory"])
        axes[1, 0].axhline(0, color="#555", linewidth=.8)
        axes[1, 0].set_title("H10 독립 선별력: 요인 통제 전후 IC")
        axes[1, 0].legend(frameon=False)
        axes[1, 0].tick_params(axis="x", rotation=20)

    if alpha_summary is not None and not alpha_summary.empty:
        aa = alpha_summary.dropna(subset=["alpha_annualized_linear", "alpha_ci95_low_annualized", "alpha_ci95_high_annualized"]).copy()
        y = np.arange(len(aa))
        xerr = np.vstack([
            aa.alpha_annualized_linear - aa.alpha_ci95_low_annualized,
            aa.alpha_ci95_high_annualized - aa.alpha_annualized_linear,
        ])
        axes[1, 1].errorbar(aa.alpha_annualized_linear, y, xerr=xerr, fmt="o", color=COLORS["primary"], capsize=5)
        axes[1, 1].axvline(0, color="#555", linewidth=.8)
        axes[1, 1].set_yticks(y, aa.model)
        axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1, 1].set_title("H10 연환산 알파와 HAC 95% 신뢰구간")
        axes[1, 1].set_xlabel("신뢰구간이 0을 포함하면 양의 알파 미검증")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(.5, .5, "알파 회귀 셀 실행 후\n신뢰구간 그래프가 생성됩니다.", ha="center", va="center")

    for ax in axes.flat:
        ax.grid(alpha=.18, axis="x")
    fig.suptitle("수익성·모델 구조·강건성 가설 시각화", fontsize=19, fontweight="bold")
    fig.text(.5, .01, "H8·H9 패널은 원 가설검증 실험 결과이며, 최종 앙상블의 공통 Stable 표본 비교와 직접 혼합하지 않는다.",
             ha="center", fontsize=11, color="#555")
    fig.tight_layout(rect=[0, .03, 1, .96])
    _save(fig, run_dir / "figures" / "presentation_research_hypotheses.png")
    return {"presentation_feature_cv": feature,
            "presentation_settled_model_comparison": settled_models,
            "presentation_factor_ic": factor,
            "presentation_alpha_summary": alpha_summary if alpha_summary is not None else pd.DataFrame()}


def build_presentation_visuals(
    run_dir: str | Path,
    project_root: str | Path,
    experiment_tables: dict[str, pd.DataFrame] | None = None,
    alpha_summary: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Create presentation-ready, evidence-bounded figures and their audit tables."""
    _setup_style()
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    supplied = experiment_tables or {}

    result: dict[str, pd.DataFrame] = {}
    _cv_locked_final_figure(run_dir, supplied)
    result["presentation_ensemble_comparison"] = _ensemble_figure(run_dir, supplied)
    result.update(_stability_figure(run_dir, project_root))
    result.update(_research_figure(run_dir, supplied, alpha_summary))
    coverage = pd.DataFrame([
        (1, "시각화", "위험축 산점도와 축 제거 실험", "결합이 변동성 단독보다 우월한지는 별도 판단"),
        (2, "시각화", "Lookback별 미래 Risky 오분류율", "252일은 평가시점이 18개라 120일과 표본 수가 다름"),
        (3, "시각화", "K별 평균 Silhouette", "K=3의 해석 가능성과 통계적 최적성을 구분"),
        (4, "시각화", "Stable/Middle/Risky 미래 변동성·MDD", "20개 평가시점의 기술통계"),
        (5, "시각화", "섹터별 Stable 비중", "정적 섹터 매핑 사용"),
        (6, "제외", "Momentum120·RSI56 단독/결합 분리 실험 없음", "현재 BASE 포함만으로 개별 효과를 식별할 수 없음"),
        (7, "시각화", "기술·거시 추가 피처의 내부 CV IC", "서로 다른 실험군은 절대수준 직접 비교 주의"),
        (8, "시각화", "완전 정산 모델의 OOS 누적수익", "4개 기간뿐이므로 일반화 주장 불가"),
        (9, "8과 통합", "학습 방식과 모델 구조가 H8과 중복", "독립 가설로 중복 집계하지 않음"),
        (10, "시각화", "요인 통제 전후 IC와 알파 신뢰구간", "알파 신뢰구간이 0을 포함"),
    ], columns=["hypothesis", "visualization_status", "evidence", "limitation"])
    result["presentation_hypothesis_coverage"] = coverage

    for name, table in result.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(run_dir / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")

    axis_corr = result["presentation_stability_axis_correlation"].iloc[0].spearman_vol_abs_beta
    axis_summary = result["presentation_stability_axis_summary"].set_index("policy")
    lookback = result["stability_lookback"].set_index("lookback")
    k_summary = result["presentation_stability_k_summary"].set_index("k")
    groups = result["stability_future_groups"].set_index("stability_group")
    ensemble = result["presentation_ensemble_comparison"].set_index("strategy")
    report = f"""# 발표용 시각화 가이드

## 생성 그래프

- `presentation_ensemble_comparison.png`: 동일한 4개 OOS 기간의 앙상블·단독 모델 수익률과 MDD. CV 고정 최종 후보와 OOS 기반 탐색 결과를 색으로 분리한다.
- `presentation_cv_locked_final_model.png`: CV에서 비중을 고정한 최종 후보만의 Fold IC, 기간수익, 모델 비중, ElasticNet 직접 비교.
- `presentation_stability_hypotheses.png`: 가설 1~5의 위험축, Lookback, K, 미래 위험, 섹터 편향.
- `presentation_research_hypotheses.png`: 가설 7, 8·9, 10의 내부 CV, 완전 정산 OOS, 요인 통제 IC, 알파 신뢰구간.

## 핵심 수치와 해석

- 최종 CV 고정 4모델은 동일 4개 OOS 구간에서 누적수익률 {ensemble.loc['cv_locked_relu_4model', 'cumulative_return']:.2%}, MDD {ensemble.loc['cv_locked_relu_4model', 'mdd']:.2%}다. 동적 ReLU의 {ensemble.loc['dynamic4_relu', 'cumulative_return']:.2%}는 OOS 성과를 비중 산정에 사용한 탐색 결과라 확증 성과로 사용하지 않는다.
- H1의 최신 단면에서 변동성과 절대 베타의 Spearman 상관은 {axis_corr:.2f}로 높다. 두 축이 독립적이라는 표현은 지지되지 않는다. 미래 Risky 오분류율은 변동성 단독 {axis_summary.loc['volatility_only', 'false_stable_rate']:.1%}, 결합 {axis_summary.loc['canonical_2d_kmeans', 'false_stable_rate']:.1%}, 베타 단독 {axis_summary.loc['beta_only', 'false_stable_rate']:.1%}로 결합이 변동성 단독을 개선하지 않았다.
- H2의 120일 오분류율은 {lookback.loc[120, 'false_stable_rate']:.1%}로, 동일 20개 평가시점을 가진 20·30·60·120일 중 가장 낮다. 252일은 {lookback.loc[252, 'false_stable_rate']:.1%}지만 평가시점이 {int(lookback.loc[252, 'dates'])}개여서 120일의 {int(lookback.loc[120, 'dates'])}개와 직접적인 최적성 비교에는 제약이 있다.
- H3 평균 Silhouette는 K=2가 {k_summary.loc[2, 'silhouette']:.3f}, K=3이 {k_summary.loc[3, 'silhouette']:.3f}다. K=3은 군집품질 최댓값이 아니라 Stable/Middle/Risky 해석을 위한 정책 선택이다.
- H4 미래 변동성은 Stable {groups.loc['Stable', 'mean_future_vol']:.1%}, Risky {groups.loc['Risky', 'mean_future_vol']:.1%}; 미래 |MDD|는 각각 {abs(groups.loc['Stable', 'mean_future_mdd']):.1%}, {abs(groups.loc['Risky', 'mean_future_mdd']):.1%}로 Stable의 위험이 일관되게 낮다.
- H10 알파 신뢰구간은 0을 포함하므로 통계적으로 양의 알파가 검증됐다고 표현하지 않는다.

## 제외·통합 원칙

가설 6은 Momentum(120d)과 RSI(56d)를 각각 단독·결합한 통제 실험이 현재 파일에 없으므로 제외했다. H8과 H9는 같은 학습 방식·모델 구조 실험을 가리켜 하나로 통합했다. H8·H9 패널은 원 가설검증 실험이며 최종 앙상블의 공통 Stable 표본 결과와 직접 혼합하지 않는다. 주황색 OOS 동적 앙상블은 성과가 높더라도 OOS 선택 편향이 있는 탐색 결과이며 최종 확증 결과로 제시하지 않는다.
"""
    (run_dir / "PRESENTATION_VISUALS.md").write_text(report, encoding="utf-8")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["presentation_visuals"] = {
            "generated": True,
            "excluded_hypotheses": [6],
            "merged_hypotheses": {"9": 8},
            "interpretation_rule": "CV-locked confirmatory and OOS-selected exploratory are visually separated",
        }
        manifest["figure_files"] = sorted(str(p.relative_to(run_dir)).replace("\\", "/") for p in (run_dir / "figures").glob("*.png"))
        manifest["table_files"] = sorted(str(p.relative_to(run_dir)).replace("\\", "/") for p in (run_dir / "tables").glob("*.csv"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    gc.collect()
    return result
