"""여러 모델의 종목 기여도 / 업종 편중 / 시장 베타 / 국면별 성과 진단.

필수: numpy, pandas, scipy. 모델 재학습이나 데이터 다운로드를 하지 않습니다.
입력 계약
----------
predictions: {'LGBM': df, ...}, 각 df는 Date, Ticker, pred. 모델 간 공통 행만 비교.
prices: Date, Ticker, Close. 배당/분할을 일관되게 반영한 수정주가 또는 총수익 지수.
market: Date, market_return(일별 *단순* 수익률), 선택 rf(일별 단순 무위험수익률).
sectors: {Ticker: Sector} 또는 Ticker, Sector DataFrame.
         Date도 있으면 그 날짜부터 유효한 PIT 업종 이력으로 backward 조회.
         Date가 없으면 현재 업종의 과거 적용일 수 있어 메타데이터에 명시.

run_model_diagnostics(...) 반환값은 DataFrame들의 dict. 수익률은 소수(0.1=10%).
기본 entry_lag=1: 신호 다음 시장 거래일 종가에 진입. 0은 신호일 종가 체결 가정.
120일마다 리밸런싱 / 120일 보유. 중간 비중은 가격에 따라 변하며 매일 재조정하지 않음.
round_trip_bps는 기간 시작 자본 대비 고정 왕복비용이며 출구에 차감. 전체동일가중/시장 기준은 비용 전.
결측 가격을 미래 정보로 걸러내거나 자동 전방채움하지 않음. 필요한 가격이 없으면 오류.
입력 예측행 자체가 미래 타깃 유무로 필터링되었거나 생존종목만 포함하면 이 함수로 복구 불가.

해석
----
concentration: 연결된 초기자본 기준 종목 기여도. 양의 기여 비율은 양의 기여 합계가 분모.
sector_weights: 리밸런싱 시점 선택 종목 비중 - 같은 시점 비교 유니버스 비중.
beta_summary: 일별 전략 순초과수익 = alpha + beta * 시장초과수익, OLS + Bartlett HAC.
              alpha는 시장 하나만 통제한 절편이며 인과적 실력의 증명이 아님.
regime_ic: 신호일 당시 시장 추세/변동성별 미래 IC. 중첩타깃이므로 날짜수는 독립 표본수가 아님.
regime_performance: 리밸런싱 신호일 국면별 *보유기간* 수익률의 평균. 국면별 가상 누적수익률이 아님.

호출 예시
---------
result = run_model_diagnostics(
    {"LGBM": te_lgbm, "XGB": te_xgb, "CatBoost": te_catboost},
    prices=price_df, market=market_df, sectors=sector_df,
    horizon=120, entry_lag=1, top_fraction=0.2, round_trip_bps=10)
print(result["comparison"])
print(result["concentration"].groupby("model").head(10))
print(result["sector_summary"])
print(result["beta_summary"])
print(result["regime_ic"])
print(result["regime_performance"])

HAC 참고: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html
CAPM의 시장초과수익/무위험수익 구분 참고: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""
from __future__ import annotations
import math
from collections.abc import Mapping
import numpy as np
import pandas as pd
from scipy.stats import norm


def _dated(frame, name, keys):
    d = frame.copy()
    missing = set(keys) - set(d.columns)
    if missing:
        raise ValueError(f'{name}: 필요한 컬럼 {sorted(missing)}')
    d['Date'] = pd.to_datetime(d['Date'])
    if d['Date'].dt.tz is not None:
        d['Date'] = d['Date'].dt.tz_localize(None)
    if not d['Date'].eq(d['Date'].dt.normalize()).all():
        raise ValueError(f'{name}: Date는 시간 없는 거래일이어야 합니다.')
    if d[list(keys)].isna().any().any():
        raise ValueError(f'{name}: 필수 컬럼에 결측값이 있습니다.')
    if 'Ticker' in keys:
        d['Ticker'] = d['Ticker'].astype(str)
    idcols = ['Date', 'Ticker'] if 'Ticker' in keys else ['Date']
    if d.duplicated(idcols).any():
        raise ValueError(f'{name}: {idcols} 중복행이 있습니다.')
    return d.sort_values(idcols).reset_index(drop=True)


def build_market_regimes(market, trend_window=126, vol_window=20,
                         vol_history=126, high_vol_quantile=.75):
    """과거 자료만 사용. vol 기준은 이전일까지 expanding 분위수. 예측 당일 종가까지 가용 가정."""
    m = _dated(market, 'market', ['Date', 'market_return'])
    r = m.set_index('Date')['market_return'].astype(float)
    if not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('market_return은 -1보다 큰 유한한 일별 단순수익률이어야 합니다.')
    trend = np.expm1(np.log1p(r).rolling(trend_window, min_periods=trend_window).sum())
    vol = r.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    cutoff = vol.expanding(min_periods=vol_history).quantile(high_vol_quantile).shift(1)
    regime = pd.Series('Unknown', index=r.index)
    known = trend.notna() & vol.notna() & cutoff.notna()
    regime.loc[known] = (
        np.where(trend.loc[known] >= 0, 'Up', 'Down') + pd.Series(
            np.where(vol.loc[known] > cutoff.loc[known], '_HighVol', '_NormalVol'),
            index=trend.loc[known].index)
    )
    return pd.DataFrame({'market_trend': trend, 'market_vol': vol,
                         'past_vol_cutoff': cutoff, 'regime': regime}).rename_axis('Date').reset_index()


def fit_market_beta(daily, hac_lag=20):
    """일별 순수익률의 CAPM 회귀. normal p/CI는 점근근사, RF 미제공시 0을 명시적으로 사용."""
    cols = ['portfolio_net', 'market_return', 'rf']
    v = daily[cols].astype(float)
    if not np.isfinite(v.to_numpy()).all():
        raise ValueError('CAPM 입력에 결측/무한대가 있습니다.')
    n = len(v)
    result = dict(n_days=n, alpha_daily=np.nan, alpha_ann_arithmetic=np.nan,
                  beta=np.nan, alpha_hac_t=np.nan, alpha_p_normal=np.nan,
                  alpha_ann_ci_low=np.nan, alpha_ann_ci_high=np.nan,
                  beta_ci_low=np.nan, beta_ci_high=np.nan, r_squared=np.nan,
                  hac_lag=min(hac_lag, max(0, n-1)), status='insufficient_data')
    if n < 30:
        return result
    x = (v.market_return - v.rf).to_numpy()
    y = (v.portfolio_net - v.rf).to_numpy()
    X = np.column_stack([np.ones(n), x])
    if np.linalg.matrix_rank(X) < 2:
        result['status'] = 'constant_market'
        return result
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coef
    scores = X * residual[:, None]
    meat = scores.T @ scores
    lag = result['hac_lag']
    for k in range(1, lag + 1):
        cross = scores[k:].T @ scores[:-k]
        meat += (1-k/(lag+1)) * (cross + cross.T)
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ meat @ bread * n/(n-2)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    a, b = coef
    t = a/se[0] if se[0] > 1e-15 else np.nan
    total_ss = np.square(y-y.mean()).sum()
    result.update(alpha_daily=a, alpha_ann_arithmetic=a*252, beta=b,
                  alpha_hac_t=t, alpha_p_normal=2*norm.sf(abs(t)) if np.isfinite(t) else np.nan,
                  alpha_ann_ci_low=(a-1.96*se[0])*252, alpha_ann_ci_high=(a+1.96*se[0])*252,
                  beta_ci_low=b-1.96*se[1], beta_ci_high=b+1.96*se[1],
                  r_squared=1-np.square(residual).sum()/total_ss if total_ss > 0 else np.nan,
                  status='ok')
    return result


def _sector_lookup(sectors):
    if sectors is None:
        return lambda date, tickers: np.repeat('Unknown', len(tickers)), 'missing'
    if isinstance(sectors, Mapping):
        mapping = {str(k): str(v) for k, v in sectors.items() if pd.notna(v)}
        return lambda date, tickers: np.array([mapping.get(t, 'Unknown') for t in tickers]), 'static'
    s = sectors.copy()
    if not {'Ticker', 'Sector'} <= set(s):
        raise ValueError('sectors에는 Ticker, Sector가 필요합니다.')
    if s['Ticker'].isna().any():
        raise ValueError('sectors: Ticker 결측값')
    s['Ticker'] = s['Ticker'].astype(str)
    s['Sector'] = s['Sector'].fillna('Unknown').astype(str)
    if 'Date' not in s:
        if s.Ticker.duplicated().any():
            raise ValueError('정적 업종표는 종목별 한 행이어야 합니다.')
        mapping = s.set_index('Ticker').Sector.to_dict()
        return lambda date, tickers: np.array([mapping.get(t, 'Unknown') for t in tickers]), 'static'
    s = _dated(s, 'sector_history', ['Date', 'Ticker', 'Sector'])
    by = {t: g.set_index('Date').Sector for t, g in s.groupby('Ticker')}
    def lookup(date, tickers):
        ans = []
        for ticker in tickers:
            history = by.get(ticker)
            pos = -1 if history is None else history.index.searchsorted(date, side='right') - 1
            ans.append('Unknown' if pos < 0 else history.iloc[pos])
        return np.array(ans)
    return lookup, 'point_in_time'


def run_model_diagnostics(predictions, prices, market, sectors=None, *,
                          pred_col='pred', price_col='Close', horizon=120,
                          entry_lag=1, offset=0, n_periods=None, top_fraction=.2,
                          round_trip_bps=10, top_k=5, beta_window=252,
                          beta_min_periods=126, hac_lag=20, min_stocks=10,
                          trend_window=126, vol_window=20, vol_history=126,
                          high_vol_quantile=.75):
    """동일 날짜/종목 교집합을 사용해 모든 모델을 진단. 자세한 계약은 모듈 docstring 참고.

    필수 반환표: comparison, concentration, holdings, period_concentration,
      sector_summary, sector_weights, beta_summary, regime_ic, regime_performance.
    추가: daily_portfolio, periods, daily_ic, regimes, alignment, metadata.
    n_periods=None이면 종료가격이 확보된 모든 비중첩 보유기간. offset은 실제 시장 거래일 단위.
    market은 최소 beta_window + 예측기간 + horizon 만큼의 기간을 제공하는 것을 권장.
    """
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError('predictions는 모델명:예측DataFrame 딕셔너리여야 합니다.')
    for name, value in dict(horizon=horizon, entry_lag=entry_lag, offset=offset,
                            top_k=top_k, beta_window=beta_window, beta_min_periods=beta_min_periods,
                            hac_lag=hac_lag, min_stocks=min_stocks).items():
        if not isinstance(value, (int, np.integer)) or value < (0 if name in ['entry_lag','offset','hac_lag'] else 1):
            raise ValueError(f'{name}: 유효하지 않은 정수')
    if not 0 < top_fraction < 1 or not 0 <= round_trip_bps < 10000:
        raise ValueError('top_fraction은 (0,1), 비용은 [0,10000) bp 범위여야 합니다.')
    if beta_min_periods > beta_window:
        raise ValueError('beta_min_periods <= beta_window 필요')
    if n_periods is not None and (not isinstance(n_periods, (int,np.integer)) or n_periods < 1):
        raise ValueError('n_periods는 양의 정수 또는 None')
    m = _dated(market, 'market', ['Date', 'market_return'])
    rf_assumption = 'provided' if 'rf' in m else 'zero_assumed'
    if 'rf' not in m:
        m['rf'] = 0.0
    if not np.isfinite(m[['market_return','rf']].to_numpy(dtype=float)).all() or (m[['market_return','rf']] <= -1).any().any():
        raise ValueError('시장/무위험 수익률은 유한한 일별 단순수익률이며 -1보다 커야 합니다.')
    m = m.set_index('Date')
    calendar = m.index
    p = _dated(prices, 'prices', ['Date', 'Ticker', price_col])
    if (p[price_col] < 0).any() or not np.isfinite(p[price_col].to_numpy(dtype=float)).all():
        raise ValueError('가격은 유한한 0 이상 값이어야 합니다. 매수가격은 양수여야 합니다.')
    price = p.pivot(index='Date', columns='Ticker', values=price_col).reindex(calendar)
    score = {}; common = None; alignment = []
    for name, frame in predictions.items():
        if not isinstance(name, str):
            raise ValueError('모델명은 문자열이어야 합니다.')
        v = _dated(frame, name, ['Date','Ticker',pred_col])
        if not np.isfinite(v[pred_col].to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: 예측값에 무한대가 있습니다.')
        sr = v.set_index(['Date','Ticker'])[pred_col]
        score[name] = sr
        common = sr.index if common is None else common.intersection(sr.index)
        alignment.append(dict(model=name, input_rows=len(v)))
    common = common.sort_values()
    if len(common) == 0:
        raise ValueError('모델 간 공통 예측행이 없습니다.')
    bydate = pd.DataFrame(index=common).reset_index().groupby('Date', sort=True)
    signal_dates = pd.DatetimeIndex(sorted(bydate.groups))
    pos = calendar.get_indexer(signal_dates)
    if (pos < 0).any():
        raise ValueError('일부 신호일이 market 거래일 달력에 없습니다.')
    valid_dates = signal_dates[pos + entry_lag + horizon < len(calendar)]
    if not len(valid_dates):
        raise ValueError('보유기간 종료일까지 시장 데이터가 확보된 신호일이 없습니다.')
    valid_set = set(valid_dates)
    eval_index = common[common.get_level_values('Date').isin(valid_dates)]
    for a in alignment:
        a.update(common_rows=len(common), evaluated_rows=len(eval_index), dropped_noncommon=a['input_rows']-len(common),
                 excluded_unmatured_common=len(common)-len(eval_index))
    sector_at, sector_mode = _sector_lookup(sectors)
    regimes = build_market_regimes(m.reset_index(), trend_window, vol_window, vol_history, high_vol_quantile).set_index('Date')
    returns = price.pct_change(fill_method=None).replace([np.inf,-np.inf], np.nan)
    xr = m.market_return - m.rf
    # 동일한 pairwise 유효일에서 공분산/시장분산 계산
    asset_excess = returns.sub(m.rf, axis=0)
    hist_beta = pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
    for ticker in price:
        y = asset_excess[ticker]; paired_x = xr.where(y.notna())
        cov = y.rolling(beta_window, min_periods=beta_min_periods).cov(paired_x)
        var = paired_x.rolling(beta_window, min_periods=beta_min_periods).var()
        hist_beta[ticker] = cov / var.where(var > 0)
    schedule_pos = np.arange(calendar.get_loc(valid_dates[0])+offset,
                             calendar.get_loc(valid_dates[-1])+1, horizon)
    if n_periods is not None:
        if len(schedule_pos) < n_periods:
            raise ValueError('요청한 n_periods만큼 완결 보유기간이 없습니다.')
        schedule_pos = schedule_pos[:n_periods]
    if not len(schedule_pos):
        raise ValueError('offset 이후 유효 리밸런싱 날짜가 없습니다.')
    scheduled = set(calendar[schedule_pos])
    if not scheduled <= valid_set:
        raise ValueError('고정 거래일 리밸런싱 날짜에 공통 예측이 없습니다. 날짜를 건너뛰지 않습니다.')
    cost = round_trip_bps/10000.0
    period_rows=[]; holding_rows=[]; sector_rows=[]; daily_rows=[]; ic_rows=[]; concentration_rows=[]
    nav={name:1.0 for name in score}
    for date in valid_dates:
        members = bydate.get_group(date)['Ticker'].astype(str).to_numpy()
        if len(members) < min_stocks:
            raise ValueError(f'{date.date()}: 공통 종목 수 {len(members)} < min_stocks')
        if not set(members) <= set(price.columns):
            raise ValueError(f'{date.date()}: 예측 종목의 가격이 없습니다.')
        signal_pos = calendar.get_loc(date); entry = signal_pos+entry_lag; end=entry+horizon
        endpoints = price.iloc[[entry,end]][members].to_numpy(dtype=float)
        if not np.isfinite(endpoints).all() or (endpoints[0] <= 0).any():
            raise ValueError(f'{date.date()}: 진입/종료 가격 누락. 종목을 사후 제외하지 말고 데이터를 보완하세요.')
        forward = endpoints[1]/endpoints[0]-1
        labels = sector_at(date, members)
        regime = regimes.loc[date,'regime']
        if date in scheduled:
            path = price.iloc[entry:end+1][members].to_numpy(dtype=float)
            if not np.isfinite(path).all():
                raise ValueError(f'{date.date()}: 보유 중 가격 누락. 자동 전방채움하지 않습니다.')
            relative = path/path[0]
            eq_path = relative.mean(axis=1)
            if (eq_path <= 0).any():
                raise ValueError('비교 유니버스 자산가치가 0 이하입니다.')
            r_m = m.market_return.iloc[entry+1:end+1].to_numpy(dtype=float)
            r_rf = m.rf.iloc[entry+1:end+1].to_numpy(dtype=float)
            market_h = np.prod(1+r_m)-1; rf_h=np.prod(1+r_rf)-1
            beta_values = hist_beta.loc[date,members].to_numpy(dtype=float)
        for name, sr in score.items():
            pred = sr.loc[date].reindex(members).to_numpy(dtype=float)
            ranked = pd.Series(pred).rank(method='average')
            ic = ranked.corr(pd.Series(forward).rank(method='average'))
            # 동률은 Ticker 오름차순으로 처리. members는 공통 Index의 Ticker 순서.
            top_n = max(1, math.ceil(len(members)*top_fraction))
            selected = np.argsort(pred, kind='stable')[-top_n:]
            top_flag = np.zeros(len(members),dtype=bool);top_flag[selected]=True
            ic_rows.append(dict(model=name,Date=date,regime=regime,n=len(members),ic=ic,
                                top_forward_gross=forward[selected].mean(),universe_forward_gross=forward.mean()))
            if date not in scheduled:
                continue
            gross_path = relative[:,selected].mean(axis=1)
            net_path = gross_path.copy();net_path[-1] -= cost
            if (net_path <= 0).any():
                raise ValueError(f'{name} {date}: 비용 적용 후 자산가치가 0 이하입니다.')
            daily_net = net_path[1:]/net_path[:-1]-1
            top_gross=forward[selected].mean();top_net=top_gross-cost;eq=forward.mean()
            beta_selected=beta_values[selected];coverage=np.isfinite(beta_selected).mean()
            beta_pre=float(beta_selected.mean()) if coverage==1 else np.nan
            positive = np.clip(forward[selected]/top_n,0,None)
            winners = selected[np.argsort(forward[selected],kind='stable')[::-1]]
            k=min(top_k,top_n);keep=winners[k:]
            ex_post_removed=float(forward[keep].mean()-cost) if len(keep) else np.nan
            concentration_rows.append(dict(model=name,Date=date,n_selected=top_n,k_removed=k,
                top_k_positive_share=float(np.sort(positive)[-k:].sum()/positive.sum()) if positive.sum()>0 else np.nan,
                top_k_return_contribution=forward[winners[:k]].sum()/top_n,
                return_without_top_k_ex_post=ex_post_removed,universe_return=eq))
            for j in selected:
                holding_rows.append(dict(model=name,Date=date,entry_date=calendar[entry],exit_date=calendar[end],
                    Ticker=members[j],Sector=labels[j],initial_weight=1/top_n,stock_return=forward[j],
                    period_contribution=forward[j]/top_n,linked_contribution=nav[name]*forward[j]/top_n,
                    beta_at_signal=beta_values[j]))
            for sector in sorted(set(labels)):
                u=(labels==sector);t=u & top_flag;wp=t.sum()/top_n;wu=u.mean()
                ru=forward[u].mean();rt=forward[t].mean() if t.any() else np.nan
                sector_rows.append(dict(model=name,Date=date,Sector=sector,top_weight=wp,universe_weight=wu,
                    active_weight=wp-wu,top_sector_return=rt,universe_sector_return=ru,
                    top_return_contribution=forward[t].sum()/top_n,
                    allocation_effect=(wp-wu)*ru,selection_effect=wp*(rt-ru) if t.any() else 0.0))
            period_rows.append(dict(model=name,Date=date,entry_date=calendar[entry],exit_date=calendar[end],regime=regime,
                n_universe=len(members),n_selected=top_n,portfolio_gross=top_gross,portfolio_net=top_net,
                universe_return=eq,market_return=market_h,excess_vs_universe=top_net-eq,
                excess_vs_market=top_net-market_h,beta_at_signal=beta_pre,beta_coverage=coverage,
                beta_adjusted_period_residual=top_net-rf_h-beta_pre*(market_h-rf_h),
                linked_cost=nav[name]*cost,unknown_sector_weight=np.mean(labels[selected]=='Unknown')))
            for j,trade_date in enumerate(calendar[entry+1:end+1]):
                daily_rows.append(dict(model=name,Date=trade_date,signal_date=date,entry_regime=regime,
                    portfolio_net=daily_net[j],universe_return=eq_path[j+1]/eq_path[j]-1,
                    market_return=r_m[j],rf=r_rf[j]))
            nav[name] *= 1+top_net
    holdings=pd.DataFrame(holding_rows);periods=pd.DataFrame(period_rows);daily=pd.DataFrame(daily_rows)
    sector_weights=pd.DataFrame(sector_rows);daily_ic=pd.DataFrame(ic_rows)
    period_concentration=pd.DataFrame(concentration_rows)
    concentration=[];sector_summary=[];beta_summary=[];comparison=[]
    for name in score:
        h=holdings[holdings.model==name];pmod=periods[periods.model==name];dd=daily[daily.model==name]
        if dd.Date.duplicated().any():
            raise AssertionError('일별 전략 수익률이 중복되었습니다.')
        grouped=h.groupby('Ticker').agg(linked_contribution=('linked_contribution','sum'),
                                        selection_count=('Date','size'),mean_initial_weight=('initial_weight','mean')).reset_index()
        pos_sum=grouped.linked_contribution.clip(lower=0).sum();abs_sum=grouped.linked_contribution.abs().sum()
        grouped['positive_contribution_share']=grouped.linked_contribution.clip(lower=0)/pos_sum if pos_sum>0 else np.nan
        grouped['absolute_contribution_share']=grouped.linked_contribution.abs()/abs_sum if abs_sum>0 else np.nan
        grouped['selection_frequency']=grouped.selection_count/len(pmod);grouped['model']=name
        concentration.append(grouped.sort_values('linked_contribution',ascending=False))
        sw=sector_weights[sector_weights.model==name]
        all_sectors=sorted(sw.Sector.unique());period_dates=pmod.Date.tolist()
        expanded=sw.set_index(['Date','Sector']).reindex(pd.MultiIndex.from_product([period_dates,all_sectors],names=['Date','Sector']))
        for sector in all_sectors:
            s=expanded.xs(sector,level='Sector')
            w=s.top_weight.fillna(0);a=s.active_weight.fillna(0)
            sector_summary.append(dict(model=name,Sector=sector,mean_weight=w.mean(),max_weight=w.max(),
                holding_frequency=(w>0).mean(),mean_active_weight=a.mean(),overweight_frequency=(a>1e-12).mean()))
        capm=fit_market_beta(dd,hac_lag);capm['model']=name;beta_summary.append(capm)
        curve=np.r_[1.,np.cumprod(1+dd.portfolio_net.to_numpy())]
        cum=curve[-1]-1;eqcum=np.prod(1+dd.universe_return)-1;mcum=np.prod(1+dd.market_return)-1
        # 연결된 기여도 - 연결된 비용 == 최종 순수익률 (초기자본 1)
        if not np.isclose(h.linked_contribution.sum()-pmod.linked_cost.sum(),cum,rtol=1e-8,atol=1e-10):
            raise AssertionError('종목 기여도와 최종 순수익률이 일치하지 않습니다.')
        icv=daily_ic[daily_ic.model==name].ic.dropna();std=icv.std()
        positive_sorted=grouped.linked_contribution.clip(lower=0).sort_values(ascending=False)
        comparison.append(dict(model=name,n_periods=len(pmod),n_daily_returns=len(dd),daily_ic=icv.mean(),
            ic_ir=icv.mean()/std if std>0 else np.nan,positive_ic_ratio=(icv>0).mean(),
            total_return_net=cum,universe_total_gross=eqcum,market_total_gross=mcum,
            excess_total_vs_universe=cum-eqcum,excess_total_vs_market=cum-mcum,
            mdd=np.min(curve/np.maximum.accumulate(curve)-1),
            top_k_positive_contribution_share=positive_sorted.head(top_k).sum()/pos_sum if pos_sum>0 else np.nan,
            mean_unknown_sector_weight=pmod.unknown_sector_weight.mean(),
            mean_beta_at_signal=pmod.beta_at_signal.mean(),beta_full_coverage_ratio=(pmod.beta_coverage==1).mean(),
            capm_beta=capm['beta'],alpha_ann_arithmetic=capm['alpha_ann_arithmetic'],alpha_p_normal=capm['alpha_p_normal']))
    regime_ic=daily_ic.groupby(['model','regime']).agg(n_dates=('Date','size'),valid_ic_days=('ic','count'),
        daily_ic=('ic','mean'),ic_std=('ic','std'),positive_ic_ratio=('ic',lambda s:(s.dropna()>0).mean()),
        mean_top_forward=('top_forward_gross','mean'),mean_universe_forward=('universe_forward_gross','mean')).reset_index()
    regime_ic['ic_ir']=regime_ic.daily_ic/regime_ic.ic_std.replace(0,np.nan)
    regime_ic['mean_forward_excess']=regime_ic.mean_top_forward-regime_ic.mean_universe_forward
    regime_performance=periods.groupby(['model','regime']).agg(n_periods=('Date','size'),
        mean_portfolio_net=('portfolio_net','mean'),mean_market=('market_return','mean'),
        mean_excess_vs_universe=('excess_vs_universe','mean'),mean_excess_vs_market=('excess_vs_market','mean'),
        win_rate_vs_universe=('excess_vs_universe',lambda s:(s>0).mean()),
        mean_beta_at_signal=('beta_at_signal','mean'),mean_beta_adjusted_residual=('beta_adjusted_period_residual','mean')).reset_index()
    metadata=pd.DataFrame([
        ('entry_assumption',f'signal + {entry_lag} market sessions, close execution'),
        ('cost_assumption',f'{round_trip_bps} bp of period initial capital, charged at exit; benchmarks gross'),
        ('sector_mode',sector_mode),('risk_free',rf_assumption),('comparison_universe','intersection of model Date/Ticker rows'),
        ('regime','signal-date trailing market trend and historical volatility threshold; NOT future market return'),
        ('regime_inference','descriptive only; 120-day IC targets overlap and period counts may be small'),
        ('beta_residual','period residual is a horizon-level approximation; daily CAPM is the regression result'),
        ('alpha_interpretation','single-market-factor residual, not causal skill or full factor alpha'),
        ('data_limits','input survivorship, prediction prefiltering, adjusted-price accuracy and executable fills not verified'),
        ('concentration','top-k removal is ex-post sensitivity, not an executable strategy'),
        ('units','decimal returns and weights; alpha_ann_arithmetic = daily intercept * 252'),
    ],columns=['item','value'])
    return dict(comparison=pd.DataFrame(comparison),concentration=pd.concat(concentration,ignore_index=True),
        holdings=holdings,period_concentration=period_concentration,
        sector_summary=pd.DataFrame(sector_summary),sector_weights=sector_weights,
        beta_summary=pd.DataFrame(beta_summary),regime_ic=regime_ic,regime_performance=regime_performance,
        daily_portfolio=daily,periods=periods,daily_ic=daily_ic,regimes=regimes.reset_index(),
        alignment=pd.DataFrame(alignment),metadata=metadata)
