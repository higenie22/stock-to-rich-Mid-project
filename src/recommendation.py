"""CV-only weights and descriptive (not causal) recommendation explanations."""
import numpy as np
import pandas as pd

MODELS = ('elasticnet', 'mlp', 'macro', 'lstm')


def relu_weights(ic):
    values = pd.Series(ic, dtype=float).reindex(MODELS)
    if not np.isfinite(values).all():
        raise ValueError('All four models require finite CV IC values')
    positive = values.clip(lower=0)
    if positive.sum() == 0:
        raise ValueError('All CV IC values are non-positive: recommendation deferred')
    return (positive / positive.sum()).to_dict()


def explain_recommendations(frame, weights):
    result = frame.copy()
    result['recommendation_rank'] = np.arange(1, len(result) + 1)
    for model, weight in weights.items():
        result[model + '_cv_weight'] = weight
    def explain(row):
        parts = [f"Stable 후보군에서 앙상블 점수 {row.pred:.4f}, 추천 {row.recommendation_rank}위. "
                 '점수는 모델별 후보군 백분위 순위의 CV ReLU 가중평균이며 기대수익률이나 상승확률이 아닙니다.']
        for col, title in [('momentum_120d', '최근 120거래일 모멘텀'),
                           ('relative_momentum_120d', '시장 대비 상대 모멘텀'),
                           ('vol_ann_120d', '120일 기준 연환산 변동성')]:
            if col in row and pd.notna(row[col]):
                parts.append(f'{title}: {row[col]:.2%}.')
        if 'beta_120d' in row and pd.notna(row.beta_120d):
            parts.append(f'시장 민감도 베타 {row.beta_120d:.3f}; 음수는 역방향 공변동을 뜻하며 안전성을 보장하지 않습니다.')
        if 'rsi_56d' in row and pd.notna(row.rsi_56d):
            parts.append(f'RSI(56일) {row.rsi_56d:.1f}/100: 최근 상승·하락 강도 지표입니다.')
        parts.append('이 설명은 관측 피처와 선정 규칙 요약이며 개별 피처의 인과적 기여도 분석은 아닙니다.')
        return ' '.join(parts)
    result['recommendation_explanation'] = result.apply(explain, axis=1)
    return result
