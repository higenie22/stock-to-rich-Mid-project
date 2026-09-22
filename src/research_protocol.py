"""제한적 실증 연구 프로토콜: 고정 설정 walk-forward + 기준전략 + 가격 경로 감사.

성과에 따라 기간/설정을 자동 선택하지 않는다. 재사용 OOS에 대한 탐색 연구이다.
데이터 파일 쓰기 없음. 경로가 불완전하면 연결 누적성과/MDD는 산출하지 않는다.
"""
import numpy as np
import pandas as pd
from . import training_scope as ts


def make_horizon_labels(frame, horizon, entry_lag=1):
    """Return an isolated, horizon-specific label panel without price filling.

    The output names contain ``horizon`` so a 20-day experiment cannot
    accidentally fit or settle against the pre-existing 120-day labels.
    Corporate-action settlement records are deliberately not transplanted to a
    different entry/exit interval; they must be supplied and verified anew.
    """
    from . import quant_validation as qv
    if horizon < 1 or entry_lag < 1:
        raise ValueError('horizon/entry_lag must be positive')
    # The caller supplies a unique panel; avoid duplicating and consolidating
    # hundreds of MiB before adding horizon-specific label columns.
    d = frame.copy(deep=False)
    if d.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Date/Ticker key duplicate')
    suffix = f'_{horizon}d_next_close'
    target = 'target_excess_simple' + suffix
    simple = 'target_simple' + suffix
    log = 'target_ret' + suffix
    market_target = 'target_market' + suffix
    entry, end = 'target_entry' + suffix, 'target_end' + suffix
    if not (d.groupby('Date')['market_return'].nunique(dropna=False) == 1).all():
        raise ValueError('market_return must be unique per Date')
    calendar = pd.DatetimeIndex(d.Date.unique()).sort_values()
    date_map = pd.Series(calendar, index=calendar)
    d[entry] = d.Date.map(date_map.shift(-entry_lag))
    d[end] = d.Date.map(date_map.shift(-(entry_lag+horizon)))
    lookup = d[['Date', 'Ticker', 'Close']].set_index(['Date', 'Ticker'])['Close']
    entry_keys = pd.MultiIndex.from_arrays([d[entry], d.Ticker], names=['Date','Ticker'])
    end_keys = pd.MultiIndex.from_arrays([d[end], d.Ticker], names=['Date','Ticker'])
    entry_close = lookup.reindex(entry_keys).to_numpy()
    exit_close = lookup.reindex(end_keys).to_numpy()
    d[log] = np.log(exit_close / entry_close)
    market = d[['Date', 'market_return']].drop_duplicates('Date').set_index('Date').market_return.astype(float)
    market_cum = market.reindex(calendar).cumsum()
    d[market_target] = d[end].map(market_cum) - d[entry].map(market_cum)
    # repair_labels turns a verified -100% event into a usable simple label;
    # no 120-day manual settlement is reused for a 20/60-day interval.
    d = qv.repair_labels(d, entry_col=entry, end_col=end, log_col=log,
                         simple_col=simple, market_col=market_target,
                         target_col=target, settlements=None)
    if 'identity_unresolved' in d:
        d.loc[d.identity_unresolved, [log, simple, target]] = np.nan
        d.loc[d.identity_unresolved, 'label_status'] = 'security_or_venue_unresolved'
    d['horizon_mature'] = d[end].notna()
    valid = np.isfinite(d[target])
    if valid.any() and not ((d.loc[valid, entry] > d.loc[valid, 'Date']) &
                            (d.loc[valid, end] > d.loc[valid, entry])).all():
        raise AssertionError('future label date contract violated')
    return d, calendar, dict(horizon=horizon, target=target, simple=simple,
                             log=log, market=market_target, entry=entry, end=end)


def capped_baseline(g, n, kind):
    """TopN 기준전략도 50% 업종 제한과 공통 피처 모집단을 사용한다."""
    a=g.copy()
    a['_baseline']=-a['vol_ann_120d']
    if kind=='lowvol': a['stable']=True
    elif kind!='stable_lowvol': raise ValueError(kind)
    # 안정군이 부족할 때는 낮은 변동성 순 상위 분위로 보충.
    return ts.choose_top(a,'_baseline',n)


def settle_path(picked, calendar, prices, entry, end, target_return, requested_n):
    """고정 초기 동일가중 buy-and-hold 경로. 종가 단면 누락을 숨기지 않는다."""
    out=dict(selected=len(picked),status='insufficient_candidates',return_settled=np.nan,
             mdd=np.nan,missing_weight=np.nan,path_coverage=np.nan,
             assumption_flat=np.nan,assumption_loss50=np.nan,assumption_loss100=np.nan)
    if len(picked)!=requested_n or requested_n==0: return out,None
    r=picked[target_return].to_numpy(float)
    known=np.isfinite(r)
    if (r[known]<-1).any(): raise ValueError('수익률 -100% 미만')
    out['missing_weight']=float((~known).mean())
    for name,penalty in [('flat',0),('loss50',-.5),('loss100',-1)]:
        out['assumption_'+name]=float((r[known].sum()+(~known).sum()*penalty)/requested_n)
    if known.all(): out['return_settled']=float(r.mean())
    dates=calendar[(calendar>=entry)&(calendar<=end)]
    if not len(dates) or dates[0]!=entry or dates[-1]!=end:
        out['status']='missing_calendar';return out,None
    block=prices.reindex(index=dates,columns=picked.Ticker.tolist())
    valid=np.isfinite(block.to_numpy())&(block.to_numpy()>0)
    out['path_coverage']=float(valid.mean())
    if not known.all(): out['status']='unresolved_return';return out,None
    if not valid.all(): out['status']='unresolved_path';return out,None
    endpoints=block.iloc[-1].to_numpy()/block.iloc[0].to_numpy()-1
    if not np.allclose(endpoints,r,atol=1e-6,rtol=1e-5):
        # 외부 정산값은 있어도 중간 가격 경로와 불일치하면 MDD를 만들지 않음.
        out['status']='endpoint_mismatch';return out,None
    wealth=block.div(block.iloc[0]).mean(axis=1)
    out['mdd']=float((wealth/wealth.cummax()-1).min())
    out['status']='settled_path'
    return out,wealth


def summarize_paths(periods,paths):
    rows=[]
    for (strategy,n),g in periods.groupby(['strategy','top_n']):
        g=g.sort_values('Date')
        row=dict(strategy=strategy,top_n=n,n_periods=len(g),settled_periods=int(g.return_settled.notna().sum()),
                 path_periods=int(g.status.eq('settled_path').sum()),
                 mean_period_return_available=g.return_settled.mean(),
                 cumulative_return=np.nan,mdd=np.nan,annualized_volatility=np.nan,
                 mdd_15_preference='unresolved',status='incomplete')
        # 기간 종료 수익률 확정과 일별 위험 경로 확정을 분리한다.
        contiguous=True
        if {'EntryDate','ExitDate'}.issubset(g.columns):
            contiguous=(g.EntryDate.iloc[1:].to_numpy()==g.ExitDate.iloc[:-1].to_numpy()).all()
        if contiguous and g.return_settled.notna().all():
            row['cumulative_return']=float((1+g.return_settled).prod()-1)
        complete=g.status.eq('settled_path').all()
        if complete:
            segments=[];capital=1.;last=None
            for p in g.itertuples():
                w=paths[(strategy,n,p.Date)]
                if last is not None and w.index[0]!=last:
                    complete=False;row['cumulative_return']=np.nan;break
                part=w*capital
                segments.append(part if last is None else part.iloc[1:])
                capital=float(part.iloc[-1]);last=part.index[-1]
            if complete:
                series=pd.concat(segments)
                if series.index.has_duplicates: raise ValueError('자산곡선 날짜 중복')
                row.update(mdd=float((series/series.cummax()-1).min()),
                           annualized_volatility=float(series.pct_change().dropna().std()*np.sqrt(252)),status='complete_path')
                row['mdd_15_preference']='met_observed_path' if row['mdd']>=-.15 else 'not_met'
        rows.append(row)
    return pd.DataFrame(rows)


def run_walkforward(frame,raw_prices,calendar,sectors,estimator,features,stability_features,
                    target,return_col,entry_col,end_col,start_date='2024-07-01',horizon=120,
                    stride=5,seed=42,k=4,progress=print):
    """매 리밸런싱 시점까지 확정된 라벨로 두 모델과 공통 군집을 재학습한다."""
    calendar=pd.DatetimeIndex(calendar).sort_values()
    if calendar.has_duplicates or horizon<1 or stride<1: raise ValueError('달력/기간/stride 오류')
    positions=np.arange(calendar.searchsorted(pd.Timestamp(start_date)),len(calendar),horizon)
    positions=positions[positions+1+horizon<len(calendar)]
    if not len(positions): raise ValueError('완전히 성숙한 평가 기간 없음')
    s=sectors.rename(columns={'ticker':'Ticker','sector':'Sector'})[['Ticker','Sector']]
    if s.Ticker.duplicated().any(): raise ValueError('업종 키 중복')
    mapping=s.set_index('Ticker').Sector.to_dict()
    if raw_prices.duplicated(['Date','Ticker']).any(): raise ValueError('가격 중복')
    panel=raw_prices.pivot(index='Date',columns='Ticker',values='Close')
    logs,periods,holdings,paths=[],[],[],{}
    for fold,pos in enumerate(positions,1):
        day,entry,end=calendar[pos],calendar[pos+1],calendar[pos+1+horizon]
        train=frame.loc[frame.Date.lt(day)&frame[end_col].lt(day)&frame.label_available_at.lt(day)&frame.eligible_signal].copy()
        dates=pd.DatetimeIndex(train.Date.unique()).sort_values()[::stride]
        train=train.loc[train.Date.isin(dates)]
        test=frame.loc[frame.Date.eq(day)&frame.eligible_signal].copy()
        if not test[entry_col].eq(entry).all() or not test[end_col].eq(end).all():
            raise ValueError('프로토콜과 타깃 체결/보유기간 불일치')
        progress(f'Walk-forward {fold}/{len(positions)}: {day.date()} 학습 시작',flush=True)
        r=ts.compare_training_scope(train,test,estimator,features,stability_features,target,end_col,k,seed,1)
        candidates=r['predictions'].copy()
        candidates['Sector']=candidates.Ticker.map(mapping).fillna('Unknown').replace({'':'Unknown','nan':'Unknown'})
        for _,metric in r['metrics'].iterrows():
            logs.append(dict(fold=fold,Date=day,max_train_label_end=train[end_col].max(),
                max_label_available=train.label_available_at.max(),stable_cluster=r['stable_cluster'],**metric.to_dict()))
        for n in [10,20]:
            selections={
                'lowvol':capped_baseline(candidates,n,'lowvol'),
                'stable_lowvol':capped_baseline(candidates,n,'stable_lowvol'),
                'all_train':ts.choose_top(candidates,'pred_all_train',n),
                'stable_train':ts.choose_top(candidates,'pred_stable_train',n)}
            for name,picked in selections.items():
                metrics,w=settle_path(picked,calendar,panel,entry,end,return_col,n)
                periods.append(dict(Date=day,EntryDate=entry,ExitDate=end,strategy=name,top_n=n,
                    fallback_n=int((~picked.stable).sum()) if name!='lowvol' else 0,**metrics))
                if w is not None: paths[(name,n,day)]=w
                h=picked.copy();h['strategy']=name;h['top_n']=n;h['initial_weight']=1/n
                h['selection_complete']=len(h)==n;holdings.append(h)
        # 군집 전체 동일가중은 별도의 진단 기준. 업종 제한을 충족하지 않으면 성과 보류.
        picked=candidates.loc[candidates.stable].copy()
        sector_ok=len(picked)>0 and picked.Sector.value_counts(normalize=True).max()<=.5
        metrics,w=settle_path(picked,calendar,panel,entry,end,return_col,len(picked))
        if not sector_ok:
            metrics.update(status='sector_constraint',return_settled=np.nan,mdd=np.nan,
                           assumption_flat=np.nan,assumption_loss50=np.nan,assumption_loss100=np.nan);w=None
        periods.append(dict(Date=day,EntryDate=entry,ExitDate=end,strategy='stable_EW',top_n=0,fallback_n=0,**metrics))
        if w is not None:paths[('stable_EW',0,day)]=w
        progress(f'Walk-forward {fold}: 공통 후보 {len(candidates)}, 안정군 {len(picked)}',flush=True)
    p=pd.DataFrame(periods)
    return dict(periods=p,summary=summarize_paths(p,paths),paths=paths,
                holdings=pd.concat(holdings,ignore_index=True),fold_log=pd.DataFrame(logs),
                protocol=dict(start_date=str(start_date),horizon=horizon,entry_lag=1,stride=stride,
                  seed=seed,k=k,features=list(features),stability_features=list(stability_features),
                  model_parameters=estimator.get_params(),cost_bps=0,sector_cap=.5,
                  limitations=['reused_OOS_exploratory','static_sector','vendor_adjusted_price_basis',
                               'missing_securities_and_settlements','four_or_few_holding_periods']))


def plot_research(result):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    periods=result['periods']
    fig,axes=plt.subplots(2,2,figsize=(15,8))
    for j,n in enumerate([10,20]):
        for strategy in ['lowvol','stable_lowvol','all_train','stable_train']:
            a=periods.loc[periods.top_n.eq(n)&periods.strategy.eq(strategy)]
            axes[0,j].plot(a.Date,a.return_settled,'o-',label=strategy)
            axes[1,j].plot(a.Date,a.mdd,'o-',label=strategy)
        axes[0,j].set_title(f'Top{n}: settled period returns (no costs)')
        axes[1,j].set_title(f'Top{n}: within-period daily-path MDD')
        axes[1,j].axhline(-.15,color='red',ls='--',label='15% preference, not guarantee')
        for ax in axes[:,j]:
            ax.axhline(0,color='black',lw=.6);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Missing results remain gaps; no imputed daily path. See summary for whole-period MDD.')
    fig.autofmt_xdate();fig.tight_layout();plt.show()


def plot_horizon_comparison(summary):
    """Compact descriptive plot for distinct 20/60/120-day experiments.

    Bars are not a common-time alpha test: the number and dates of mature
    periods differ by horizon. Missing complete paths remain visibly absent.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    a=summary.loc[summary.top_n.eq(10) & summary.strategy.isin(['all_train','stable_train'])].copy()
    fig, axes=plt.subplots(1, 2, figsize=(13,4))
    labels=[]
    for _,r in a.iterrows():
        labels.append(f"{int(r.horizon)}d\n{r.strategy.replace('_train','')}")
    axes[0].bar(range(len(a)),a.mean_period_return_available)
    axes[0].axhline(0,color='black',lw=.7)
    axes[0].set_xticks(range(len(a)),labels)
    axes[0].set_title('Top10: observed-period mean return')
    axes[0].set_ylabel('conditional simple return')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for i,r in enumerate(a.itertuples()):
        axes[0].text(i,0,f'{r.settled_periods}/{r.n_periods}',ha='center',va='bottom',fontsize=8)
    m=a.mdd.copy()
    axes[1].bar(range(len(a)),m)
    axes[1].axhline(0,color='black',lw=.7);axes[1].axhline(-.15,color='red',ls='--',lw=.8)
    axes[1].set_xticks(range(len(a)),labels)
    axes[1].set_title('Top10: whole-path MDD (blank = incomplete)')
    axes[1].set_ylabel('MDD')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for i,v in enumerate(m):
        if not np.isfinite(v): axes[1].text(i,0,'incomplete',ha='center',va='bottom',fontsize=8)
    fig.suptitle('20/60/120-day targets are different prediction tasks; no automatic winner selection')
    fig.tight_layout();plt.show()


def paired_scope_summary(by_horizon):
    """Compare the two fitting populations only on identical observed periods.

    This is a complete-case descriptive comparison, not an investable return:
    periods with an unresolved selected constituent are omitted as a *pair*.
    The omitted count is reported rather than silently treated as zero.
    """
    rows=[]
    for horizon,result in by_horizon.items():
        p=result['periods']
        x=p.loc[p.top_n.eq(10)&p.strategy.isin(['all_train','stable_train']),
                ['Date','strategy','return_settled']].pivot(index='Date',columns='strategy',values='return_settled')
        complete=x.dropna(subset=['all_train','stable_train'])
        rows.append(dict(horizon=horizon, scheduled_periods=len(x),
            paired_observed_periods=len(complete),
            paired_coverage=len(complete)/len(x) if len(x) else np.nan,
            all_train_mean=complete.all_train.mean(),
            stable_train_mean=complete.stable_train.mean(),
            stable_minus_all_mean=(complete.stable_train-complete.all_train).mean(),
            stable_wins=int((complete.stable_train>complete.all_train).sum()),
            all_wins=int((complete.stable_train<complete.all_train).sum())))
    return pd.DataFrame(rows)
    # 일별 자산곡선은 전 기간 경로가 완전한 전략만 연결한다.
    fig,axes=plt.subplots(1,2,figsize=(15,4))
    for ax,n in zip(axes,[10,20]):
        shown=0
        for strategy in ['lowvol','stable_lowvol','all_train','stable_train']:
            state=result['summary'].loc[result['summary'].top_n.eq(n)&result['summary'].strategy.eq(strategy)]
            if state.empty or state.iloc[0].status!='complete_path':continue
            parts=[];capital=1.
            for day in periods.loc[periods.top_n.eq(n)&periods.strategy.eq(strategy),'Date'].sort_values():
                w=result['paths'][(strategy,n,day)]*capital
                parts.append(w if not parts else w.iloc[1:]);capital=float(w.iloc[-1])
            wealth=pd.concat(parts)
            ax.plot(wealth.index,wealth-1,label=strategy);shown+=1
        ax.set_title(f'Top{n}: complete daily wealth paths only')
        if shown:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'No complete path: see period-level results',ha='center',transform=ax.transAxes)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();plt.show()
