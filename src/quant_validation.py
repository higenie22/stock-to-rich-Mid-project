"""시간·유니버스·정산·선택편향 검증. 파일 다운로드/원본 수정/자동 운용 승인 없음.

핵심 원칙
1. 티커는 영구 증권 식별자가 아니다. PIT 자료가 없으면 보정 완료로 주장하지 않는다.
2. 결측 정답은 학습할 수 없지만 검증 후보/포트폴리오에서 사후 제거하지 않는다.
3. 120일 중첩은 독립 표본 수를 정확히 n/120으로 만든다는 뜻이 아니다.
4. 잔차 IC는 통제변수에 조건부인 기술통계이며 거래 가능한 알파의 증명이 아니다.
5. 모델 선택까지 포함한 외부 fold 평가와 선택에 재사용한 CV 점수를 구분한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from scipy.stats import norm


SIVB_HALT_SOURCE = 'https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-halts-svb-financial-group'


def universe_contract(frame, pit=None):
    """신호일의 거래 후보를 표시한다. 가격행 자체는 삭제하지 않는다.

    pit 입력: Date, Ticker, SecurityID, eligible, known_at, source의 일별 완전 스냅샷.
    해당 신호일까지 알려진 행만 허용한다. 입력 스냅샷의 외부 정확성/완전성은 별도 감사.
    제공됐는데 해당 날짜/티커 행이 없으면 unknown으로 분류하고 후보로 허용하지 않는다.
    과거 구성 여부를 현재 구성종목 명단에서 역산하거나 첫/마지막 가격일로 대체하지 않는다.
    """
    d = frame.copy()
    d['Date'] = pd.to_datetime(d['Date'])
    d['universe_status'] = 'unverified_history'
    d['eligible_signal'] = True  # 연구용만 허용. 이 상태는 승인 관문에서 차단된다.
    d['SecurityID'] = pd.NA
    notes = []
    if pit is not None:
        required = ['Date', 'Ticker', 'SecurityID', 'eligible', 'known_at', 'source']
        if not set(required) <= set(pit):
            raise ValueError(f'PIT 스냅샷 필요 컬럼: {required}')
        p = pit[required].copy()
        p['Date'], p['known_at'] = pd.to_datetime(p['Date']), pd.to_datetime(p['known_at'])
        if p[required].isna().any().any() or p.duplicated(['Date', 'Ticker']).any():
            raise ValueError('PIT 스냅샷 필수값 누락/중복')
        if not p['eligible'].isin([True, False, 0, 1]).all():
            raise ValueError('eligible은 bool/0/1이어야 함 (문자열은 명시 변환 필요)')
        if (p['known_at'] > p['Date']).any() or p['source'].astype(str).str.strip().eq('').any():
            raise ValueError('신호일 이후에 알려진 PIT 행/출처 없는 행')
        d = d.drop(columns='SecurityID').merge(p.rename(columns={'eligible': '_eligible'}),
                                               on=['Date', 'Ticker'], how='left', validate='one_to_one')
        matched = d['SecurityID'].notna()
        d['eligible_signal'] = matched & d['_eligible'].eq(True)
        d['universe_status'] = np.where(matched, 'snapshot_supplied', 'missing_snapshot')
        d.drop(columns='_eligible', inplace=True)
        notes.append('PIT 스냅샷 연결됨. 출처 자체의 정확성은 외부 검증 필요')
    else:
        notes.append('PIT 스냅샷 없음: 현재 결과는 유니버스 편향 미보정 연구 결과')
    # Nasdaq는 2023-03-10 개장 전 SIVB 거래정지를 공지했다. 이후 OTC 이력이 같은
    # 티커로 병합돼 있을 수 있으므로 거래소/영구ID가 없는 자료에서 신규 편입을 금지한다.
    # 과거 보유 포지션을 현금 0 또는 -100%로 임의 정산하지 않는 것이 중요하다.
    blocked = d['Ticker'].eq('SIVB') & d['Date'].ge(pd.Timestamp('2023-03-10'))
    d.loc[blocked, 'eligible_signal'] = False
    d.loc[blocked, 'universe_status'] = 'SIVB_venue_identity_unresolved'
    if 'Volume' in d:
        no_trade = ~np.isfinite(pd.to_numeric(d['Volume'], errors='coerce')) | d['Volume'].le(0)
        d.loc[no_trade, 'eligible_signal'] = False
        d.loc[no_trade & ~blocked, 'universe_status'] = 'no_signal_day_volume'
    notes.append(f'SIVB 신호 차단 {int(blocked.sum()):,}행; 근거 {SIVB_HALT_SOURCE}')
    return d, pd.DataFrame({'note': notes})


def repair_labels(d, *, entry_col, end_col, log_col, simple_col, market_col, target_col,
                  settlements=None):
    """종목식별이 불명확한 구간은 정답 미확정. 확인된 정산만 개별 보유기간에 연결.

    settlements: Date,Ticker,EntryDate,ExitDate,simple_return,available_at,verified,source.
    simple_return은 현금/교환주식/분배금 등을 포함하여 '진입~예정 청산일' 전체 기간의
    검증된 단순수익률이어야 한다. 인수 발표 프리미엄이나 마지막 거래가를 그대로 넣으면 안 됨.
    수동 verified 플래그는 출처의 진위를 자동 증명하지 않는다. 원래 관측값과 충돌하면 오류.
    """
    # Only horizon-label columns are added or changed. A shallow copy keeps the
    # immutable feature blocks shared and prevents a full-panel duplication.
    out = d.copy(deep=False)
    out['label_available_at'] = pd.to_datetime(out[end_col])
    out[simple_col] = np.expm1(out[log_col])
    out['label_status'] = np.where(out[end_col].isna(), 'unmatured',
                                  np.where(np.isfinite(out[simple_col]), 'observed', 'missing_price'))
    # SIVB 정지일을 가로지르는 포지션: 정지 이후 가격을 정상 Nasdaq 체결로 간주하지 않음.
    ambiguous = out['Ticker'].eq('SIVB') & out[end_col].ge(pd.Timestamp('2023-03-10'))
    if 'SecurityID' in out and out['SecurityID'].notna().any():
        lookup = out.set_index(['Date', 'Ticker'])['SecurityID']
        for col in [entry_col, end_col]:
            keys = pd.MultiIndex.from_arrays([out[col], out['Ticker']])
            ids = pd.Series(lookup.reindex(keys).to_numpy(), index=out.index)
            ambiguous |= out[end_col].notna() & (ids.isna() | out['SecurityID'].isna() | ids.ne(out['SecurityID']))
    out.loc[ambiguous, simple_col] = np.nan
    out.loc[ambiguous, 'label_status'] = 'security_or_venue_unresolved'
    out['settlement_source'] = pd.NA
    if settlements is not None and len(settlements):
        cols = ['Date', 'Ticker', 'EntryDate', 'ExitDate', 'simple_return', 'available_at', 'verified', 'source']
        if not set(cols) <= set(settlements):
            raise ValueError(f'정산 입력 필요 컬럼: {cols}')
        s = settlements[cols].copy()
        for col in ['Date', 'EntryDate', 'ExitDate', 'available_at']:
            s[col] = pd.to_datetime(s[col])
        if s[cols].isna().any().any() or s.duplicated(['Date', 'Ticker']).any():
            raise ValueError('정산 필수값 누락/중복')
        if not s['verified'].eq(True).all() or s['source'].astype(str).str.strip().eq('').any():
            raise ValueError('검증/출처 없는 정산은 사용할 수 없음')
        if not np.isfinite(s.simple_return).all() or (s.simple_return < -1).any():
            raise ValueError('정산 단순수익률은 유한하고 -1 이상이어야 함')
        if (s['available_at'] < s['ExitDate']).any():
            raise ValueError('전체 보유기간 정산은 청산일보다 먼저 확정될 수 없음')
        index = pd.MultiIndex.from_frame(out[['Date', 'Ticker']])
        positions = index.get_indexer(pd.MultiIndex.from_frame(s[['Date', 'Ticker']]))
        if (positions < 0).any():
            raise ValueError('정산 입력에 대상 예측행이 없음')
        for pos, (_, row) in zip(positions, s.iterrows()):
            key = out.index[pos]
            if out.at[key, entry_col] != row.EntryDate or out.at[key, end_col] != row.ExitDate:
                raise ValueError('정산 진입/청산일 불일치')
            old = out.at[key, simple_col]
            if np.isfinite(old) and not np.isclose(old, row.simple_return, atol=1e-8):
                raise ValueError('관측 수익률과 제공 정산이 충돌: 원가격/기업행사를 먼저 대조하세요')
            out.at[key, simple_col] = row.simple_return
            out.at[key, 'label_status'] = 'verified_settlement'
            out.at[key, 'settlement_source'] = row.source
            out.at[key, 'label_available_at'] = row.available_at
    # -100%는 log(0)=-inf이므로 회귀 학습에서 단순수익률을 사용한다.
    # 회계상 실제 단순수익률과 학습용 시장 초과수익률은 별도 열로 유지한다.
    with np.errstate(divide='ignore', invalid='ignore'):
        out[log_col] = np.log1p(out[simple_col])
    out[target_col] = out[simple_col] - np.expm1(out[market_col])
    return out


def label_coverage(frame, target_col):
    d = frame.copy()
    d['year'] = pd.to_datetime(d['Date']).dt.year
    d['label_available'] = np.isfinite(d[target_col])
    if 'label_status' not in d:
        d['label_status'] = np.where(d.label_available, 'available', 'missing')
    return d.groupby(['year', 'label_status'], dropna=False).agg(
        rows=('Ticker', 'size'), tickers=('Ticker', 'nunique'), available=('label_available', 'sum')).reset_index()


def evaluation(frame, pred_col, target_col, date_col='Date', q=5):
    """순위는 미래 라벨 결측을 보기 전에 확정. IC의 가용 표본 편향은 커버리지로 공개.

    q_spread는 양끝 바스켓이 모두 정산된 날짜만의 평균이다. 그 날짜 수도 함께 반환.
    q_spread_available은 원래 바스켓의 관측 종목 평균 차이이며 조건부 기술통계일 뿐이다.
    완전한 실제 백테스트는 별도 엔진에서 원비중으로 계산한다.
    """
    # 객체형 증권ID/출처 열까지 replace하면 pandas의 묵시적 형변환이 발생할 수 있다.
    # 평가에 필요한 숫자 열만 정리한다. 원본 프레임/타깃은 수정하지 않는다.
    v = frame[[date_col, pred_col, target_col]+(['Ticker'] if 'Ticker' in frame else [])].copy()
    for col in {pred_col, target_col}:
        v[col] = v[col].where(np.isfinite(v[col]), np.nan)
    v = v.dropna(subset=[date_col, pred_col]).sort_values([date_col]+(['Ticker'] if 'Ticker' in v else []))
    v['q'] = v.groupby(date_col)[pred_col].transform(
        lambda x: pd.qcut(x.rank(method='first'), q, labels=False)+1 if len(x) >= q else np.nan)
    rows, qs, bucket_rows = [], [], []
    for date, g in v.groupby(date_col):
        a = g.dropna(subset=[target_col])
        ic = a[pred_col].rank().corr(a[target_col].rank()) if len(a) >= 5 and a[pred_col].nunique() > 1 and a[target_col].nunique() > 1 else np.nan
        means, full = {}, {}
        for bucket in range(1, q+1):
            y = g.loc[g['q'].eq(bucket), target_col]
            means[bucket] = y.mean()
            full[bucket] = bool(len(y) and y.notna().all())
            bucket_rows.append(dict(Date=date, q=bucket, n=len(y), observed=int(y.notna().sum()),
                available_mean=means[bucket], settled_mean=means[bucket] if full[bucket] else np.nan))
        rows.append(dict(Date=date, ic=ic, n_candidates=len(g), n_evaluated=len(a),
                         q_spread=means[q]-means[1] if full[q] and full[1] else np.nan,
                         q_spread_available=means[q]-means[1]))
        if all(full.values()):
            qs.append(means)
    daily = pd.DataFrame(rows, columns=['Date', 'ic', 'n_candidates', 'n_evaluated', 'q_spread', 'q_spread_available']).set_index('Date')
    ic = daily['ic'].dropna()
    valid = v.dropna(subset=[target_col])
    std = ic.std()
    report = dict(daily_ic=float(ic.mean()), ic_std=float(std),
        ic_ir=float(ic.mean()/std) if std > 0 else np.nan,
        hit_rate=float((ic > 0).mean()) if len(ic) else np.nan,
        q_spread=float(daily.q_spread.mean()), q_spread_available=float(daily.q_spread_available.mean()),
        q_spread_settled_days=int(daily.q_spread.notna().sum()),
        overall_ic=float(valid[pred_col].rank().corr(valid[target_col].rank())) if len(valid) > 1 else np.nan,
        n_days=len(ic), n_obs=len(valid), n_input=len(frame), n_missing_evaluation=len(frame)-len(valid),
        evaluation_coverage=len(valid)/len(frame) if len(frame) else np.nan)
    qm = pd.DataFrame(qs).reindex(columns=range(1, q+1)).mean()
    qm.index.name = 'q'
    ic.attrs['daily_evaluation'] = daily
    buckets = pd.DataFrame(bucket_rows)
    if len(buckets):
        stats = buckets.groupby('q').agg(available_mean=('available_mean', 'mean'),
            settled_mean=('settled_mean', 'mean'), settled_days=('settled_mean', 'count'),
            observed=('observed', 'sum'), candidates=('n', 'sum'))
        stats['coverage'] = stats.observed / stats.candidates
        ic.attrs['quintile_summary'] = stats
    return report, ic, qm


def dependence_report(series, calendar, horizon=120, n_boot=1000, seed=42, min_blocks=10):
    """HAC와 이동 블록 부트스트랩 민감도. 소수 블록에서는 p/CI를 산출하지 않는다.

    min_blocks=10은 프로젝트의 사전 지정 검증 보류 기준이지 독립성 보증 정리가 아니다.
    n/horizon은 기간 길이 지표일 뿐 유효 표본 수의 정답이 아니다. HAC n_eff 역시
    관측 자기공분산/대역폭에 의존하는 진단값이고 독립 관측치 개수로 단정할 수 없다.
    날짜 결측은 달력에 재색인하여 시간 간격을 유지한다. 부트스트랩은 달력 결측 시 보류.
    """
    if series.empty:
        return pd.DataFrame([dict(status='no_data', n_obs=0)])
    s = series.sort_index()
    cal = pd.DatetimeIndex(calendar)
    cal = cal[(cal >= s.index.min()) & (cal <= s.index.max())]
    x = s.reindex(cal).to_numpy(dtype=float)
    finite = np.isfinite(x); n = int(finite.sum())
    mean = float(np.nanmean(x)) if n else np.nan
    rows = []
    rng = np.random.default_rng(seed)
    for block in sorted({max(2, horizon//2), horizon, 2*horizon}):
        count = len(x)//block
        row = dict(block_sessions=block, span_sessions=len(x), n_obs=n, full_blocks=count,
                   mean=mean, hac_t_exploratory=np.nan, effective_n_hac_diagnostic=np.nan,
                   block_ci_low=np.nan, block_ci_high=np.nan, block_p_two_sided=np.nan,
                   status='insufficient_blocks' if count < min_blocks else 'conditional_inference')
        if n > 2:
            z = np.where(finite, x-mean, 0.0)
            lag = min(block, len(z)-1)
            lrv = np.dot(z, z)/n
            for k in range(1, lag+1):
                lrv += 2*(1-k/(lag+1))*np.dot(z[k:], z[:-k])/n
            if lrv > 0:
                row['hac_t_exploratory'] = mean/np.sqrt(lrv/n*n/(n-1))
                row['effective_n_hac_diagnostic'] = float(np.clip(n*np.nanvar(x, ddof=1)/lrv, 1, n))
        if not finite.all():
            row['status'] = 'calendar_gaps_in_metric'
        elif count >= min_blocks and n > 2 and np.std(x) > 0:
            samples = np.empty(n_boot)
            draws = int(np.ceil(n/block))
            for b in range(n_boot):
                starts = rng.integers(0, n-block+1, size=draws)
                sample = np.concatenate([x[start:start+block] for start in starts])[:n]
                samples[b] = sample.mean()
            centered = samples-mean  # H0: mean=0로 중심화한 블록 평균 분포
            row['block_p_two_sided'] = float((1+(np.abs(centered) >= abs(mean)).sum())/(n_boot+1))
            low, high = np.quantile(centered, [.025, .975])
            row['block_ci_low'], row['block_ci_high'] = mean-high, mean-low
        rows.append(row)
    return pd.DataFrame(rows)


def incremental_factor_ic(frame, target_col, controls=('momentum_120d', 'momentum_252d', 'beta_120d'), sectors=None):
    """같은 날짜의 예측·타깃 순위에서 통제 피처 순위/섹터 더미를 각각 제거한 부분 IC.

    예측 잔차는 예측+당일 피처만 사용하므로 미래 y를 보지 않는다. y 잔차는 사후 진단용이며
    학습 입력/매매 점수에 넣지 않는다. 기본 학습 타깃도 이 함수에서 바꾸지 않는다.
    섹터는 Date/known_at 이력이 있는 경우만 사용. 현재 정적 섹터를 과거에 소급하지 않는다.
    """
    names = [c for c in controls if c in frame]
    if len(names) != len(controls):
        return pd.DataFrame([dict(status='missing_controls', reason=str(sorted(set(controls)-set(names))))]), pd.DataFrame()
    d = frame.copy()
    sector_mode = 'not_controlled_no_PIT'
    if sectors is not None and {'Date', 'Ticker', 'Sector', 'known_at'} <= set(sectors):
        p = sectors[['Date', 'Ticker', 'Sector', 'known_at']].copy()
        p['Date'], p['known_at'] = pd.to_datetime(p.Date), pd.to_datetime(p.known_at)
        if p.duplicated(['Date', 'Ticker']).any() or (p.known_at > p.Date).any():
            raise ValueError('PIT 섹터 중복/미래 known_at')
        d = d.merge(p[['Date', 'Ticker', 'Sector']], on=['Date', 'Ticker'], how='left', validate='one_to_one')
        sector_mode = 'PIT_sector'
    rows, residual_rows = [], []
    for date, g in d.groupby('Date'):
        # 변환용 후보는 미래 y의 존재와 무관하게 고정한다.
        a = g.copy()
        for col in ['pred', target_col]+names:
            a[col] = a[col].where(np.isfinite(a[col]), np.nan)
        a = a.dropna(subset=['pred']+names)
        if sector_mode == 'PIT_sector':
            a = a.dropna(subset=['Sector'])
        if len(a) < 20:
            continue
        X = a[names].rank(pct=True).to_numpy(dtype=float)
        if sector_mode == 'PIT_sector':
            X = np.column_stack([X, pd.get_dummies(a.Sector, drop_first=True, dtype=float).to_numpy()])
        X = np.column_stack([np.ones(len(a)), X])
        pr = a['pred'].rank(pct=True).to_numpy()
        score_residual = pr-X@np.linalg.lstsq(X, pr, rcond=None)[0]
        # 모멘텀 자체인 예측을 중립화하면 0에 가까운 부동소수점 오차만 남을 수 있다.
        # 이 오차를 다시 순위화해 가짜 거래 신호를 만들지 않는다.
        if np.std(score_residual) <= 1e-10:
            score_residual[:] = np.nan
        residual_rows.append(pd.DataFrame({'Date': date, 'Ticker': a.Ticker,
                                          'pred_neutralized': score_residual}).reset_index(drop=True))
        valid = np.isfinite(a[target_col].to_numpy())
        if valid.sum() < max(20, X.shape[1]+5):
            continue
        Z = X[valid]
        y = a.loc[valid, target_col].rank(pct=True).to_numpy()
        p = a.loc[valid, 'pred'].rank(pct=True).to_numpy()
        ry = y-Z@np.linalg.lstsq(Z, y, rcond=None)[0]
        rp = p-Z@np.linalg.lstsq(Z, p, rcond=None)[0]
        corr = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if np.std(u) > 1e-10 and np.std(v) > 1e-10 else np.nan
        rows.append(dict(Date=date, raw_ic=corr(p, y), incremental_ic=corr(rp, ry),
            momentum_ic=corr(a.loc[valid, 'momentum_120d'].rank().to_numpy(), y),
            n_candidates=len(g), n_evaluated=int(valid.sum()), controls=','.join(names),
            sector_mode=sector_mode, status='diagnostic_only'))
    return pd.DataFrame(rows), pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()


def feature_ready_dates(frame, features):
    """후보 공통 피처의 워밍업 이후 날짜. 미래 타깃/성과는 참조하지 않는다.

    모든 종목이 완전해야 한다는 조건이 아니다. 날짜당 적어도 한 종목이
    공통 피처를 계산할 수 있는지로 초기 워밍업만 결정한다.
    """
    features = list(dict.fromkeys(features))
    ready = np.isfinite(frame[features].to_numpy(dtype=float)).all(axis=1)
    if 'eligible_signal' in frame:
        ready &= frame.eligible_signal.to_numpy(dtype=bool)
    return pd.DatetimeIndex(frame.loc[ready, 'Date'].unique()).sort_values()


def common_cv_bounds(frame, features, n_splits=5):
    """노트북의 오래된 전역 경계에 의존하지 않는 공통 워밍업 경계 생성."""
    dates = feature_ready_dates(frame, features)
    size = len(dates) // (n_splits + 1)
    if size < 1:
        raise ValueError('공통 피처 준비 후 CV 기간 부족')
    return [(dates[size*(i+1)], dates[-1] if i == n_splits-1 else dates[size*(i+2)-1])
            for i in range(n_splits)]


def make_bounds(frame, end_col, n_splits=3, features=None):
    dates = (feature_ready_dates(frame, features) if features is not None
             else pd.DatetimeIndex(frame.Date.unique()).sort_values())
    if len(dates) < 2 * n_splits:
        raise ValueError('공통 피처 워밍업 이후 CV 날짜 부족')
    # 각 inner fold가 적어도 라벨 horizon을 purge할 여유를 갖도록 앞 절반을 초기 학습으로 사용.
    start = len(dates)//2
    edges = np.linspace(start, len(dates), n_splits+1, dtype=int)
    return [(dates[edges[i]], dates[edges[i+1]-1]) for i in range(n_splits) if edges[i+1] > edges[i]]


def fixed_candidate_cv(frame, features, target_col, end_col, model, bounds, direction=1, train_stride=5):
    """각 fold 학습 시점까지 확정된 y만 fit. 검증 후보는 y 결측과 무관하게 예측한다."""
    if any(c.startswith(('target_', 'trade_', 'diagnostic_', 'settled_', 'label_', 'missing_')) for c in features):
        raise ValueError('미래 정답/정산/감사 컬럼이 피처에 포함됨')
    d = frame.sort_values(['Date', 'Ticker']).copy()
    if 'eligible_signal' in d:
        d = d.loc[d['eligible_signal']].copy()
    for col in list(dict.fromkeys(features+[target_col])):
        d[col] = d[col].where(np.isfinite(d[col]), np.nan)
    d = d.dropna(subset=features)
    predictions, fold_rows = [], []
    for fold, (start, stop) in enumerate(bounds, 1):
        available_at = d['label_available_at'] if 'label_available_at' in d else d[end_col]
        tr = d.loc[(d.Date < start) & (d[end_col] < start) & (available_at < start)]
        va = d.loc[(d.Date >= start) & (d.Date <= stop)].copy()
        available = tr[target_col].notna()
        label_missing = int((~available).sum())
        tr = tr.loc[available]
        dates = pd.DatetimeIndex(tr.Date.unique()).sort_values()[::train_stride]
        tr = tr.loc[tr.Date.isin(dates)]
        if tr.Date.nunique() < 10 or va.empty:
            raise ValueError(f'purge 후 학습/검증 기간 부족: fold={fold}, '
                             f'validation={start}~{stop}, features={len(features)}, '
                             f'train_dates={tr.Date.nunique()}, train_rows={len(tr)}, '
                             f'validation_rows={len(va)}, stride={train_stride}')
        assert tr[end_col].max() < va.Date.min()
        m = clone(model).fit(tr[features], tr[target_col])
        va['pred'] = direction*m.predict(va[features])
        va['fold'] = fold
        rep, _, _ = evaluation(va, 'pred', target_col)
        rep.update(fold=fold, train_start=tr.Date.min(), train_end=tr.Date.max(),
                   max_train_label_end=tr[end_col].max(), validation_start=va.Date.min(),
                   validation_end=va.Date.max(), train_rows=len(tr), train_dates=tr.Date.nunique(),
                   train_stride=train_stride, train_missing_labels=label_missing)
        fold_rows.append(rep)
        predictions.append(va)
    out = pd.concat(predictions, ignore_index=True)
    rep, _, _ = evaluation(out, 'pred', target_col)
    return rep, out, pd.DataFrame(fold_rows)


def selection_with_outer_cv(frame, feature_sets, target_col, end_col, model, param_dist,
                            final_bounds, n_candidates=6, inner_splits=3, outer_splits=3,
                            direction=1, train_stride=5, seed=42, run_outer=True):
    """같은 후보 집합/선택 규칙을 내부와 최종 선택에 사용. 외부 정답은 선택에 전달하지 않음.

    모델 종류 간 승자를 외부 점수로 고르는 것도 다시 선택이다. 독립 최종 평가는 아니다.
    CV/OOS를 보고 방향을 정하지 않는다. direction은 실험 시작 전에 고정한다.
    """
    rng = np.random.default_rng(seed)
    sets = list(feature_sets)
    candidates = [(sets[i % len(sets)], params) for i, params in enumerate(
        ParameterSampler(param_dist, n_iter=n_candidates, random_state=seed))]
    # 후보 수가 세트 수 미만이면 전 세트를 시험했다고 주장하지 않는다. 후보는 사전 고정.
    if len(candidates) < len(sets):
        chosen = rng.choice(sets, len(candidates), replace=False)
        candidates = [(str(chosen[i]), p) for i, (_, p) in enumerate(candidates)]

    # 거래량 ablation 쌍은 사전 고정된 동일 파라미터로 반드시 평가한다.
    if '120+Volume' in feature_sets and '120' in feature_sets:
        if len(candidates) < 2:
            raise ValueError('거래량 쌍 비교에는 후보 수 2 이상 필요')
        shared_params = candidates[0][1]
        candidates[:2] = [('120', shared_params), ('120+Volume', shared_params)]

    def select(data, bounds):
        results, best = [], None
        for name, params in candidates:
            candidate = clone(model).set_params(**params)
            rep, pred, folds = fixed_candidate_cv(data, feature_sets[name], target_col, end_col,
                candidate, bounds, direction=direction, train_stride=train_stride)
            results.append(dict(rep, feature_set=name, parameters=params))
            score = rep['daily_ic']  # 부호 있는 평균. fold별 사후 반전/mean(abs(IC)) 금지.
            if np.isfinite(score) and (best is None or score > best[0]):
                best = (score, name, params, pred, folds)
        if best is None:
            raise ValueError('유효한 후보 없음')
        return best, pd.DataFrame(results).sort_values('daily_ic', ascending=False).reset_index(drop=True)

    outer_predictions, outer_log = [], []
    selected_bounds = list(final_bounds)[-outer_splits:]
    if run_outer:
        for fold, (start, stop) in enumerate(selected_bounds, 1):
            available_at = frame['label_available_at'] if 'label_available_at' in frame else frame[end_col]
            past = frame.loc[(frame.Date < start) & (frame[end_col] < start) & (available_at < start)].copy()
            try:
                common_features = list(dict.fromkeys(c for fs in feature_sets.values() for c in fs))
                best, _ = select(past, make_bounds(past, end_col, inner_splits, features=common_features))
                _, name, params, _, _ = best
                rep, pred, _ = fixed_candidate_cv(frame, feature_sets[name], target_col, end_col,
                    clone(model).set_params(**params), [(start, stop)], direction=direction, train_stride=1)
                pred['outer_fold'] = fold
                outer_predictions.append(pred)
                outer_log.append(dict(fold=fold, status='evaluated', validation_start=start,
                    validation_end=stop, feature_set=name, parameters=params, daily_ic=rep['daily_ic'],
                    n_obs=rep['n_obs'], coverage=rep['evaluation_coverage']))
            except ValueError as exc:
                # 미평가 fold를 제외한 평균을 전체 기간 성공으로 취급하지 않는다. 이유/건수 반환.
                outer_log.append(dict(fold=fold, status='not_evaluated', validation_start=start,
                                      validation_end=stop, reason=str(exc), daily_ic=np.nan))
            print(f'외부 검증 {fold}/{len(selected_bounds)}: {outer_log[-1]["status"]}', flush=True)
    best, search = select(frame, final_bounds)
    score, name, params, pred, folds = best
    return dict(best_params=params, feature_set=name, feature_cols=list(feature_sets[name]),
                search=search, oof=pred, folds=folds, outer_folds=pd.DataFrame(outer_log),
                outer_predictions=pd.concat(outer_predictions, ignore_index=True) if outer_predictions else pd.DataFrame(),
                candidate_count=len(candidates), direction=direction,
                selection_score=score, outer_enabled=run_outer)
