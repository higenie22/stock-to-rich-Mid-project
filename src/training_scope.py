"""전체 학습 vs 저위험군 한정 학습의 통제 비교. 데이터 쓰기/다운로드 없음.

호출자가 정한 피처·파라미터를 양쪽에 동일 적용한다.
기본 연구 경로는 고정 설정, 탐색 부록에서는 기존 Train-CV 설정을 전달한다.
저위험군 전용 하이퍼파라미터 최적화나 nested 군집 전략 검증은 아니다.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from . import quant_validation as qv


def compare_training_scope(train, test, estimator, features, stability_features,
                           target, label_end, k=4, seed=42, direction=1):
    """OOS target은 fit/군집/추천에 사용하지 않는다. 두 모델의 예측 후보는 동일."""
    features, stability_features = list(features), list(stability_features)
    if any(c.startswith(('target_', 'trade_', 'label_', 'settled_')) for c in features+stability_features):
        raise ValueError('미래 정보 피처 금지')
    tr, te = train.copy(), test.copy()
    for name, d in [('train',tr),('test',te)]:
        if d.duplicated(['Date','Ticker']).any(): raise ValueError(name+' 키 중복')
    if 'eligible_signal' in tr: tr=tr.loc[tr.eligible_signal].copy()
    if 'eligible_signal' in te: te=te.loc[te.eligible_signal].copy()
    start=te.Date.min()
    if tr.empty or te.empty or not (tr.Date.lt(start)&tr[label_end].lt(start)).all():
        raise ValueError('Train/OOS 경계 또는 purge 오류')
    if 'label_available_at' in tr and not tr.label_available_at.lt(start).all():
        raise ValueError('OOS 시작 이전에 확정되지 않은 학습 라벨')
    ready=np.isfinite(tr[stability_features].to_numpy(float)).all(axis=1)
    if ready.sum()<max(k,20): raise ValueError('군집 학습 피처 부족')
    scaler=RobustScaler().fit(tr.loc[ready,stability_features])
    km=KMeans(n_clusters=k,random_state=seed,n_init=10).fit(scaler.transform(tr.loc[ready,stability_features]))
    centers=pd.DataFrame(km.cluster_centers_,columns=stability_features)
    # 기존 위험점수와 같은 정의. 스케일된 중심에서 MDD는 부호 반전.
    risk=centers.mul([-1 if c.startswith('mdd_') else 1 for c in stability_features],axis=1).sum(axis=1)
    stable_id=int(risk.idxmin())
    tr['stable']=False
    tr.loc[ready,'stable']=km.labels_==stable_id
    common=np.isfinite(te[list(dict.fromkeys(features+stability_features))].to_numpy(float)).all(axis=1)
    candidates=te.loc[common].sort_values(['Date','Ticker']).copy()
    if candidates.empty: raise ValueError('공통 예측 후보 없음')
    candidates['stable']=km.predict(scaler.transform(candidates[stability_features]))==stable_id
    # 공통 안정성 피처 가용 모집단에서 군집 소속만으로 학습 범위를 나눈다.
    fit_ready=ready & np.isfinite(tr[features+[target]].to_numpy(float)).all(axis=1)
    models, reports, ic_series = {}, [], {}
    for scope in ['all_train','stable_train']:
        a=tr.loc[fit_ready & (tr.stable if scope=='stable_train' else True)]
        if len(a)<50 or a.Date.nunique()<10: raise ValueError(scope+' 학습 표본 부족')
        m=clone(estimator)
        if 'allow_writing_files' in m.get_params(): m.set_params(allow_writing_files=False)
        m.fit(a[features],a[target])
        col='pred_'+scope
        candidates[col]=direction*m.predict(candidates[features])
        evaluation=candidates.loc[candidates.stable]
        if 'horizon_mature' in evaluation: evaluation=evaluation.loc[evaluation.horizon_mature]
        rep,ic,_=qv.evaluation(evaluation,col,target)
        rep.update(scope=scope,train_rows=len(a),train_dates=a.Date.nunique(),
                   common_prediction_rows=len(candidates),common_feature_coverage=len(candidates)/len(te))
        reports.append(rep);models[scope]=m;ic_series[scope]=ic
    return dict(models=models, predictions=candidates, metrics=pd.DataFrame(reports),
                scaler=scaler,kmeans=km,stable_cluster=stable_id,risk_scores=risk,
                ic=ic_series,comparison='same_parameters_different_training_population')


def choose_top(g, pred, n, sector_cap=.5):
    """저위험군 우선, 부족하면 전체 Q5 보충. 미래 정답은 읽지 않는다.

    업종 미확인은 Unknown 한 그룹으로 묶어 보수적으로 상한 적용.
    제약으로 N개를 못 채우면 임의 재가중하지 않고 incomplete로 반환한다.
    """
    a=g.sort_values([pred,'Ticker'],ascending=[False,True]).copy()
    if len(a)<5: return a.iloc[:0]
    q=pd.qcut(a[pred].rank(method='first'),5,labels=False)+1
    pool=pd.concat([a.loc[a.stable],a.loc[~a.stable & q.eq(5)]]).drop_duplicates('Ticker')
    counts,selected={},[]
    limit=int(np.floor(n*sector_cap))
    for idx,row in pool.iterrows():
        sector=row['Sector']
        if counts.get(sector,0)>=limit: continue
        selected.append(idx);counts[sector]=counts.get(sector,0)+1
        if len(selected)==n: break
    return pool.loc[selected].copy()


def recommendation_comparison(result, signal_dates, return_col, entry_col, end_col,
                              sectors=None, sizes=(10,20)):
    """비용 전 가상 동일가중 성과. 실제 사용자 선택/정수 매수 결과 아님."""
    d=result['predictions'].copy()
    if sectors is None:
        mapping={}
    else:
        s=sectors.rename(columns={'ticker':'Ticker','sector':'Sector'})[['Ticker','Sector']]
        if s.Ticker.duplicated().any(): raise ValueError('업종 티커 중복')
        mapping=s.set_index('Ticker').Sector.to_dict()
    d['Sector']=d.Ticker.map(mapping).fillna('Unknown')
    d['Sector']=d.Sector.replace({'':'Unknown','nan':'Unknown'})
    rows,holdings=[],[]
    for day in signal_dates:
        g=d.loc[d.Date.eq(day)]
        for n in sizes:
            for scope in ['all_train','stable_train']:
                picked=choose_top(g,'pred_'+scope,n)
                complete=len(picked)==n
                r=picked[return_col].to_numpy(float)
                known=np.isfinite(r)
                if (r[known]<-1).any(): raise ValueError('단순수익률 -100% 미만')
                row=dict(Date=day,scope=scope,top_n=n,selected=len(picked),
                    fallback_n=int((~picked.stable).sum()),
                    status='settled' if complete and known.all() else 'missing_return' if complete else 'insufficient_candidates',
                    return_settled=float(r.mean()) if complete and known.all() else np.nan,
                    missing_weight=float((~known).sum()/n) if complete else np.nan)
                for label,value in [('flat',0),('loss50',-.5),('loss100',-1)]:
                    row['assumption_'+label]=float((r[known].sum()+(~known).sum()*value)/n) if complete else np.nan
                row['EntryDate']=g[entry_col].iloc[0] if len(g) else pd.NaT
                row['ExitDate']=g[end_col].iloc[0] if len(g) else pd.NaT
                rows.append(row)
                picked=picked.copy();picked['scope']=scope;picked['top_n']=n
                picked['planned_weight']=1/n;picked['recommendation_complete']=complete
                holdings.append(picked)
    return pd.DataFrame(rows),pd.concat(holdings,ignore_index=True) if holdings else pd.DataFrame()


def plot_comparison(periods, model_name):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    for ax,n in zip(axes,[10,20]):
        for scope in ['all_train','stable_train']:
            a=periods.loc[periods.top_n.eq(n)&periods.scope.eq(scope)]
            ax.plot(a.Date,a.return_settled,'o-',label=scope)
        ax.axhline(0,color='black',lw=.7);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f'{model_name}: Top{n} settled period returns (before costs)')
        ax.legend();ax.grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();plt.show()
