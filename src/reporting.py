from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .stability import StabilityPolicy
from .visualization import save_figures


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_계산 가능한 표본이 없어 결과를 보류합니다._"
    shown = frame.head(max_rows).copy()
    for column in shown.select_dtypes(include="float").columns:
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [clean(column) for column in shown.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def _figure(figures: dict[str, str], key: str, caption: str) -> str:
    relative_path = figures.get(key)
    return f"![{caption}]({relative_path})" if relative_path else ""


def _stability_interpretation(tables: dict[str, pd.DataFrame]) -> str:
    group = tables.get("stability_future_groups", pd.DataFrame())
    if group.empty or "Stable" not in set(group.stability_group):
        return "미래 위험 표본이 부족해 Stable의 미래 안정성 판단을 보류한다."
    stable = group.loc[group.stability_group.eq("Stable")].iloc[0]
    others = group.loc[~group.stability_group.eq("Stable")]
    if len(others) and stable.future_risky_rate < others.future_risky_rate.mean():
        return "Stable의 미래 고위험 비율이 다른 군집 평균보다 낮아 위험 구분 방향은 관찰됐다. 인과효과나 일반화 성능으로 해석하지 않는다."
    return "Stable의 미래 고위험 비율이 다른 군집보다 일관되게 낮지 않아 현재 정의의 미래 위험 구분력을 확정하지 않는다."


def _industry_interpretation(tables: dict[str, pd.DataFrame]) -> str:
    table = tables.get("stability_industry_policy", pd.DataFrame())
    if table.empty:
        return "업종 보강 결과는 표본 부족으로 보류한다."
    indexed = table.set_index("policy")
    if {"market_stable", "balanced_verified"}.issubset(indexed.index):
        delta = indexed.at["balanced_verified", "false_stable_rate"] - indexed.at["market_stable", "false_stable_rate"]
        direction = "감소" if delta < 0 else "증가"
        return f"Balanced Verified의 False-Stable 비율은 Market Stable 대비 {delta:+.2%}p {direction}했다. 후보 유지율과 날짜 블록 CI를 함께 봐야 한다."
    return "Market Stable과 산업 보강 정책의 공통 비교표가 없어 결론을 보류한다."


def _profit_interpretation(tables: dict[str, pd.DataFrame]) -> str:
    summary = tables.get("profit_horizon_summary", pd.DataFrame())
    if summary.empty:
        return "완전히 성숙한 수익 평가기간이 없어 수익성 결론을 보류한다."
    settled = summary.loc[summary.settled_periods.gt(0)]
    if settled.empty:
        return "정산 가능한 기간이 없어 관측 수익의 비교를 보류한다."
    best = settled.sort_values("mean_period_return_available", ascending=False).iloc[0]
    return (
        f"정산 가능한 기간의 조건부 평균수익이 가장 높은 조합은 {int(best.horizon)}일/{best.strategy}다. "
        "정산 누락, 재사용 OOS, 서로 다른 기간 수 때문에 최종 우월성으로 채택하지 않는다."
    )


def _ensemble_interpretation(tables: dict[str, pd.DataFrame]) -> str:
    selection = tables.get('profit_candidate_selection', pd.DataFrame())
    if len(selection) and selection.iloc[0]['status'] == 'selection_deferred':
        return '미정산 종목 -50% 청산 가정을 적용한 뒤에도 완전한 평가 경로와 MDD -15% 기준을 충족하는 후보가 없어 선정을 보류했다.'
    summary = tables.get("profit_final_ensemble_summary", pd.DataFrame())
    if summary.empty:
        return "앙상블은 실행 가능한 표본 또는 PyTorch 환경이 부족해 결과를 보류한다."
    row = summary.iloc[0]
    if row.get("status") != "complete_path":
        return "일부 OOS 경로가 미정산 또는 불완전하여 누적수익·MDD의 최종 판단을 보류한다."
    return (
        f"선택 후보는 {row.get('strategy', 'unknown')}이다. 공통 표본 비교와 위험 제약에 따른 탐색적 선정이며, "
        "동일 OOS 기간의 수익·MDD만으로 일반적 우월성이나 실전 채택을 확정하지 않는다."
    )


def build_report(
    config: ProjectConfig,
    policy: StabilityPolicy,
    tables: dict[str, pd.DataFrame],
    run_dir: Path,
    figures: dict[str, str],
) -> str:
    sections = [
        "# S&P 500 안정군·수익성 통합 가설 검증 보고서",
        "",
        "이 보고서는 `sp500_stability_v14.ipynb`와 `test_validation_report.ipynb`의 가설을 하나의 데이터 계약과 하나의 Stable 정의로 재배치한 제출용 실행 결과다.",
        "",
        "## H0. 공통 분석 파이프라인",
        "",
        f"- 프로젝트 루트: `{config.project_root}`",
        "- 입력: `data/processed/final_df.parquet`, 시장 가격지수, 편입이력, 정적 Sector, 고정 거시시장 사본",
        f"- 통일 Stable: 날짜별 K={policy.k}, `{policy.volatility_column}` + `|{policy.beta_column}|`, 축별 표준화 후 가장 낮은 중심 군집",
        "- 미래 위험과 수익은 모두 신호일 다음 거래일 진입을 기준으로 계산한다.",
        "- 미래 위험값은 Stable 생성이나 수익 추천 입력으로 전달하지 않는다.",
        "",
        "### 데이터 커버리지",
        "",
        _markdown_table(tables.get("data_coverage", pd.DataFrame())),
        "",
        "## S-H1. 현재 Stable은 미래 위험을 구분하는가?",
        "",
        "K 후보의 내부 군집지표를 확인한 뒤 고정 K=3 정책을 미래 120거래일 변동성·MDD·상위 30% 위험 여부로 검증한다.",
        "",
        _markdown_table(tables.get("stability_k_sensitivity", pd.DataFrame()).groupby("k").mean(numeric_only=True).reset_index() if len(tables.get("stability_k_sensitivity", pd.DataFrame())) else pd.DataFrame()),
        "",
        _markdown_table(tables.get("stability_future_groups", pd.DataFrame())),
        "",
        f"**해석:** {_stability_interpretation(tables)}",
        "",
        "## S-H2. Volatility와 |Beta| 두 축은 단일 축보다 유용한가?",
        "",
        "동일 날짜·동일 후보 수에서 Volatility-only, Beta-only, 통일 2축 KMeans의 False-Stable과 미래 MDD를 비교한다.",
        "",
        _markdown_table(tables.get("stability_axis_ablation", pd.DataFrame()).groupby("policy").mean(numeric_only=True).reset_index() if len(tables.get("stability_axis_ablation", pd.DataFrame())) else pd.DataFrame()),
        "",
        "**해석:** 상관이나 군집 내부지표만으로 Beta의 추가가치를 선언하지 않고, 미래 위험 결과와 후보 유지 조건을 함께 판단한다.",
        "",
        "## S-H3. 긴 위험 Lookback이 False-Stable을 줄이는가?",
        "",
        "위험 Lookback 20/30/60/120/252를 같은 미래 120일 outcome으로 비교한다. 이는 수익 보유기간 비교와 별개의 실험이다.",
        "",
        _markdown_table(tables.get("stability_lookback", pd.DataFrame())),
        "",
        "**해석:** 120일은 통계적 최적값이 아니라 원본 정책과 수익성 handoff를 연결하기 위한 사전 고정 절충안이다.",
        "",
        "## S-H4. 산업 내 상대안정성 보강은 False-Stable을 줄이는가?",
        "",
        _markdown_table(tables.get("stability_industry_policy", pd.DataFrame())),
        "",
        _markdown_table(tables.get("stability_industry_bootstrap", pd.DataFrame())),
        "",
        f"**해석:** {_industry_interpretation(tables)}",
        "",
        "## S-H5. Sector별 미래위험 모델은 Global 모델보다 나은가?",
        "",
        "현재 위험 두 축으로 미래 위험 상위 30%를 분류하고 연속형 위험점수를 회귀한다. Global 모델은 2024 Validation에서 사전 규칙으로 고른 뒤 2025 Test에 적용하며, Sector refit은 같은 구조를 사용한다.",
        "",
        _markdown_table(tables.get("stability_risk_ml_classification", pd.DataFrame())),
        "",
        _markdown_table(tables.get("stability_risk_ml_regression", pd.DataFrame())),
        "",
        _markdown_table(tables.get("stability_risk_ml_sector", pd.DataFrame())),
        "",
        "**해석:** Classification의 Recall을 주지표로 사용하고 Regression은 동일 위험 outcome에서 파생된 보조 consistency check로만 본다. Confirmatory Sector가 2개 미만이면 Sector-wide 결론을 내리지 않는다.",
        "",
        "## 안정성 → 수익성 Handoff",
        "",
        "수익 모델에는 현재 시점의 Stable·산업 상태만 전달한다. `future_` 위험 outcome은 전달하지 않는다.",
        "",
        _markdown_table(tables.get("stability_handoff", pd.DataFrame()).head(10)),
        "",
        "## P-H1. 수익 타깃과 예측·보유기간",
        "",
        "미래 가격, 절대수익률, 시장 초과수익률 타깃을 비교하고 20/60/120일 예측·진입·보유·정산 규칙을 함께 변경한다. 위험 Lookback 120일과 수익 Horizon을 혼동하지 않는다.",
        "",
        _markdown_table(tables.get("profit_horizon_summary", pd.DataFrame())),
        "",
        _figure(figures, "horizon", "기간별 IC와 MDD 비교"),
        "",
        f"**해석:** {_profit_interpretation(tables)}",
        "",
        "## P-H2. 모멘텀 차분·거래량은 추가 정보를 주는가?",
        "",
        "통일 Stable 표본에서 기본 4피처, 차분 6개 추가, 거래량 2개 추가를 같은 외부 시점의 IC로 비교한다.",
        "",
        _markdown_table(tables.get("profit_controlled_models", pd.DataFrame()).query("case in ['base_excess_lgbm','difference_features','volume_features']") if len(tables.get("profit_controlled_models", pd.DataFrame())) else pd.DataFrame()),
        "",
        _figure(figures, "features", "피처별 외부 스냅샷 IC"),
        "",
        "**해석:** IC와 실제 Top-N 수익·낙폭은 별개의 결과다. 한 지표의 개선만으로 피처를 자동 채택하지 않는다.",
        "",
        "### P-H2-3. 거시시장 지표와 상호작용",
        "",
        "금리차·VIX·달러·유가·HYG의 지연 시장 대용지표와 종목 위험 상호작용을 동일 Stable·동일 표본에서 비교한다.",
        "",
        _markdown_table(tables.get("profit_macro_metrics", pd.DataFrame())),
        "",
        _figure(figures, "macro", "거시 및 상호작용 피처 외부 IC"),
        "",
        "**해석:** 같은 날짜의 거시 값은 횡단면에서 상수이므로 부분 IC 통제변수로 사용하지 않는다. 모델 추가 실험과 시장국면 연결 진단으로만 해석한다.",
        "",
        "## P-H3. 학습 범위와 모델 구조",
        "",
        "전체 학습과 Stable-only 학습을 비교하고, 고정 LGBM·최근 3년 rolling·LGBMRanker·ElasticNet·MLP·LSTM을 공통 Stable 정의에서 평가한다.",
        "",
        _markdown_table(tables.get("profit_controlled_models", pd.DataFrame()).query("case in ['base_excess_lgbm','rolling_3y','ranker','elasticnet','mlp']") if len(tables.get("profit_controlled_models", pd.DataFrame())) else pd.DataFrame()),
        "",
        _markdown_table(tables.get("profit_lstm_metrics", pd.DataFrame())),
        "",
        _figure(figures, "lstm", "LSTM 내부 검증과 외부 IC"),
        "",
        "### P-H3-3. 과거 내부 CV로 정한 예측 방향",
        "",
        _markdown_table(tables.get("profit_direction_cv", pd.DataFrame())),
        "",
        "**해석:** 재사용 평가기간의 고정 후보 비교이며 각 구조의 최적 성능 비교가 아니다. 수익과 낙폭이 동시에 개선되는지 확인한다.",
        "",
        "## P-H4. 통일 Stable 기반 포트폴리오는 저변동성 기준보다 나은가?",
        "",
        _markdown_table(tables.get("profit_horizon_periods", pd.DataFrame())),
        "",
        _figure(figures, "portfolio", "기간별 포트폴리오 정산 및 결측 민감도"),
        "",
        "### 타깃·피처·모델 통제실험의 Top10 정산 요약",
        "",
        _markdown_table(tables.get("profit_controlled_summary", pd.DataFrame())),
        "",
        "**해석:** 미정산이 있는 전략을 제외하고 승자를 정하지 않는다. -50%/-100% 값은 결측 민감도이지 실제 정산값이 아니다.",
        "",
        "## P-H5. 요인·시장·종목·업종 집중 강건성",
        "",
        _markdown_table(tables.get("profit_factor_robustness", pd.DataFrame())),
        "",
        _markdown_table(tables.get("profit_concentration", pd.DataFrame())),
        "",
        _figure(figures, "factor", "사후 부분 순위 IC"),
        "",
        "**해석:** 부분 IC는 사후 진단이며 정적 Sector는 시점별 분류를 보장하지 않는다. 짧은 평가기간으로 일반적인 독립 알파나 통계적 유의성을 주장하지 않는다.",
        "",
        "## P-Final. 최종 앙상블 모델",
        "",
        "120거래일 Stable-only 학습에서 ElasticNet 단독, 기본 LightGBM 단독, 두 모델 50/50, 두 모델과 LSTM 각 1/3, 기존 MLP/Macro/LSTM 40/40/20을 동일 표본으로 비교한다. Top-10은 단일 Sector 최대 5종목으로 제한한다. 미정산·불완전 경로 종목은 청산일 투자금 대비 -50%로 가정한다. 관측 가격을 유지하고 중간 결측은 직전 가격으로 연결하며 진입 가격이 없으면 청산 전까지 초기 투자금을 유지한다. 이 가정 경로의 전체 MDD가 -15% 이상인 후보 중 누적수익 최대를 선택한다. 동률은 MDD, 평균 IC, 이름 순이며 적격 후보가 없으면 보류한다. 실제 정산값은 별도 보존하고 가정 MDD는 실제 관측값이 아니다. 비용 차감 전 탐색적 선정이다.",
        "",
        _markdown_table(tables.get('profit_candidate_summary', pd.DataFrame())),
        "",
        _markdown_table(tables.get('profit_candidate_selection', pd.DataFrame())),
        "",
        _figure(figures, 'candidates', '동일 표본 최종 후보 비교'),
        "",
        "### 검증 역할과 프로토콜",
        "",
        _markdown_table(tables.get("profit_final_ensemble_protocol", pd.DataFrame())),
        "",
        "최신 추천은 2020~2022 Purged CV의 모델별 평균 IC로 계산한 ReLU 비중을 사용한다. OOS 수익률·MDD는 추천 모델이나 비중 선택에 사용하지 않는다. 기존 후보의 OOS 비교는 탐색적 결과이며, 연구 전반의 설계 선택까지 독립 검증되었다는 뜻은 아니다.",
        "",
        "### CV 기반 모델 비중",
        "",
        _markdown_table(tables.get('cv_locked_weights', pd.DataFrame())),
        "",
        "### OOS 백테스트 요약",
        "",
        _markdown_table(tables.get("profit_final_ensemble_summary", pd.DataFrame())),
        "",
        "### Fold별 예측 검증",
        "",
        _markdown_table(tables.get("profit_final_ensemble_metrics", pd.DataFrame())),
        "",
        "### 최신 신호일 Top-10 (미정산 실시간 추천)",
        "",
        _markdown_table(tables.get("profit_final_ensemble_recommendation", pd.DataFrame())),
        "",
        "### Sector 노출 (최대 50% 제약 확인)",
        "",
        _markdown_table(tables.get("profit_final_ensemble_sector_exposure", pd.DataFrame())),
        "",
        _figure(figures, "ensemble", "최종 앙상블 OOS와 최신 추천"),
        "",
        f"**해석:** {_ensemble_interpretation(tables)}",
        "",
        "## 최종 결론과 범위",
        "",
        "두 원본의 핵심 질문은 유지됐지만 Stable 생성기는 하나로 통일됐다. 결과는 위험 후보 생성의 유효성, 수익 예측의 선별력, 실제 정산 포트폴리오의 성과를 분리해 읽어야 한다. 관측 개선은 후속 검증 후보를 좁히는 근거이며 안전한 최대수익이나 일반적 알파의 증명이 아니다.",
        "",
        f"실행 산출물: `{run_dir}`",
    ]
    return "\n".join(sections) + "\n"


def save_run(
    config: ProjectConfig,
    policy: StabilityPolicy,
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> Path:
    run_dir = config.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    table_dir = run_dir / "tables"
    table_dir.mkdir()
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    figures = save_figures(tables, run_dir / "figures")
    report = build_report(config, policy, tables, run_dir, figures)
    (run_dir / "UNIFIED_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    payload = {
        **metadata,
        "figure_files": list(figures.values()),
        "stable_policy": {
            "lookback": policy.lookback,
            "k": policy.k,
            "features": [policy.volatility_column, f"abs({policy.beta_column})"],
            "fit_scope": "date-wise cross-section",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "SUCCESS.txt").write_text(
        "Unified stability and profitability validation completed.\n", encoding="utf-8"
    )
    return run_dir
