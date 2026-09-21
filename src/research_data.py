"""Phase 1: 로컬 자료만 사용. 가격 보존 / 신호 자격 분리 / 캐시 무조건 재사용 금지.

이력은 사후 복원 자료이며 영구ID·당시 공시시각을 검증한 데이터는 아니다.
시장 기준은 저장된 ^GSPC 가격지수: 배당 포함 투자 가능 벤치마크로 주장하지 않는다.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from src.get_tickers import TRUSTED_RENAMES


def _rolling_mdd(close, window, block_size=128):
    """Rolling MDD with bounded temporary memory."""
    values = np.asarray(close, dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    for start in range(0, len(windows), block_size):
        block = windows[start:start + block_size]
        peaks = np.maximum.accumulate(block, axis=1)
        result[start + window - 1:start + window - 1 + len(block)] = np.min(
            block / peaks - 1, axis=1
        )
    return result


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_membership(path):
    m = pd.read_csv(path, dtype={'ticker': 'string'})
    if m['ticker'].isna().any():
        raise ValueError('편입 티커 결측')
    m['original_ticker'] = m.ticker.str.strip().str.upper().str.replace('.', '-', regex=False)
    # 합병은 이름 변경이 아니다. WRK -> SW를 연결하지 않는다.
    aliases = {k: v for k, v in TRUSTED_RENAMES.items() if k != 'WRK'}
    m['ticker'] = m.original_ticker.replace(aliases)
    for col in ['start_date', 'end_date']:
        m[col] = pd.to_datetime(m[col], errors='raise')
    if m.start_date.isna().any() or (m.end_date.notna() & m.end_date.le(m.start_date)).any():
        raise ValueError('편입 구간 오류')
    return m


def mark_membership(frame, membership, verified_through):
    """행 삭제 없이 신호 자격만 부여. 종료일 제외. 커버리지 밖 신호는 차단."""
    # Only new eligibility columns are added below. A shallow copy prevents a
    # full consolidation/duplication of the multi-million-row feature panel.
    d = frame.copy(deep=False)
    keys = d[['Date', 'Ticker']].reset_index(drop=True).reset_index(names='_row')
    j = keys.merge(membership, left_on='Ticker', right_on='ticker', how='left')
    good = j.Date.ge(j.start_date) & (j.end_date.isna() | j.Date.lt(j.end_date))
    member = np.zeros(len(d), dtype=bool)
    member[j.loc[good, '_row'].to_numpy()] = True
    d['membership_eligible'] = member & d.Date.le(pd.Timestamp(verified_through))
    # 영구 증권 식별자를 티커로 위조하지 않는다.
    d['SecurityID'] = pd.NA
    d['identity_unresolved'] = d.Ticker.isin(['WRK', 'SW'])
    volume_ok = np.isfinite(d.Volume) & d.Volume.gt(0)
    d['eligible_signal'] = d.membership_eligible & volume_ok & ~d.identity_unresolved
    d['eligible_signal'] &= ~(d.Ticker.eq('SIVB') & d.Date.ge('2023-03-10'))
    d['universe_status'] = np.where(d.membership_eligible, 'historical_interval_matched', 'not_member_or_outside_history')
    d.loc[d.membership_eligible & ~volume_ok, 'universe_status'] = 'no_signal_day_volume'
    d.loc[d.identity_unresolved, 'universe_status'] = 'merger_identity_quarantine'
    return d


def coverage(frame, membership, verified_through):
    """가격 확보 종목이 아닌 이력상 후보를 분모로 사용한다."""
    rows = []
    for day, g in frame.groupby('Date'):
        if day > pd.Timestamp(verified_through):
            continue
        expected = set(membership.loc[membership.start_date.le(day) &
            (membership.end_date.isna() | membership.end_date.gt(day)), 'ticker'])
        actual = set(g.Ticker)
        rows.append(dict(Date=day, expected=len(expected), price_available=len(expected & actual),
                         missing=len(expected-actual), missing_tickers=','.join(sorted(expected-actual))))
    return pd.DataFrame(rows)


def build_features(price_path, benchmark_path, windows=(20, 60, 120, 252)):
    """공통 시장 달력에서 계산. 결측을 압축하거나 미래가격으로 보간하지 않는다.

    원가격 행은 모두 반환: 피처 워밍업 결측은 모델 입력 시점에만 처리한다.
    mdd는 각 창 안의 고점 대비 이후 저점으로 정확히 계산한다.
    """
    raw = pd.read_parquet(price_path)
    raw['Date'] = pd.to_datetime(raw.Date)
    if raw.duplicated(['Ticker', 'Date']).any() or not np.isfinite(raw.Close).all() or raw.Close.le(0).any():
        raise ValueError('가격 키 중복/비정상 가격')
    market = pd.read_parquet(benchmark_path).sort_values('Date').set_index('Date').Close
    if market.index.has_duplicates or market.isna().any() or market.le(0).any():
        raise ValueError('시장 달력/가격 오류')
    calendar = pd.DatetimeIndex(market.index)
    if not raw.Date.isin(calendar).all():
        raise ValueError('가격 날짜가 시장 달력 밖에 존재')
    mr = np.log(market / market.shift()).fillna(0.0)
    parts = []
    for ticker, g in raw.groupby('Ticker', sort=True):
        a = g.set_index('Date').reindex(calendar)
        close = a.Close
        ret = np.log(close / close.shift())
        a['log_return'], a['market_return'] = ret, mr
        delta = close.diff()
        # 0은 원자료 결측 대체값일 수도 있으므로 정상 관측으로 간주하지 않는다.
        volume = a['Volume'].astype(float).where(a['Volume'].gt(0))
        previous_mean = volume.shift(1).rolling(20, min_periods=20).mean()
        a['relative_volume_20d'] = volume / previous_mean - 1
        a['volume_trend_20_120'] = (volume.rolling(20, min_periods=20).mean()
                                   / volume.rolling(120, min_periods=120).mean() - 1)
        for w in windows:
            a[f'vol_ann_{w}d'] = ret.rolling(w).std() * np.sqrt(252)
            a[f'downside_vol_{w}d'] = np.sqrt(ret.clip(upper=0).pow(2).rolling(w).mean()*252)
            a[f'beta_{w}d'] = ret.rolling(w).cov(mr) / mr.rolling(w).var().replace(0, np.nan)
            a[f'mdd_{w}d'] = _rolling_mdd(close.to_numpy(), w)
            a[f'ma_gap_{w}d'] = close / close.rolling(w).mean() - 1
            a[f'momentum_{w}d'] = close / close.shift(w) - 1
            a[f'relative_momentum_{w}d'] = ret.rolling(w).sum() - mr.rolling(w).sum()
        for w in [14, 28, 56]:
            up, down = delta.clip(lower=0).rolling(w).mean(), (-delta.clip(upper=0)).rolling(w).mean()
            a[f'rsi_{w}d'] = (100-100/(1+up/down)).where(down.ne(0), 100).mask(up.eq(0)&down.eq(0), 50)
        wc = close.resample('W-FRI').last()
        change = wc.diff()
        up, down = change.clip(lower=0).rolling(14).mean(), (-change.clip(upper=0)).rolling(14).mean()
        wr = (100-100/(1+up/down)).where(down.ne(0), 100).mask(up.eq(0)&down.eq(0), 50).shift(1)
        a['rsi_14w'] = wr.reindex(calendar, method='ffill')
        for p in ['momentum', 'relative_momentum']:
            for label, x, y in [('short',20,60),('mid',60,120),('long',120,252)]:
                a[f'{p}_diff_{label}'] = a[f'{p}_{x}d'] - a[f'{p}_{y}d']
        # 전체 관측 가격 이력 반환. 미래 라벨 유무로 후보를 삭제하지 않는다.
        a = a.loc[g.Date].copy(deep=False)
        a['Ticker'] = ticker
        parts.append(a.rename_axis('Date').reset_index())
    # DataFrame.replace consolidates every numeric block and can temporarily
    # duplicate hundreds of MiB on the full panel. Clean one column at a time;
    # parts are already appended in sorted ticker/date order.
    out = pd.concat(parts, ignore_index=True)
    for col in out.select_dtypes(include='number').columns:
        values = out[col].to_numpy(copy=False)
        infinite = np.isinf(values)
        if infinite.any():
            out.loc[infinite, col] = np.nan
    return out.reset_index(drop=True)
