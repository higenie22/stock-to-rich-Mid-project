"""Common-sample candidate comparison; selection is exploratory, before costs."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits
from . import profitability as p
from . import research_protocol as protocol
from .recommendation import MODELS, relu_weights, explain_recommendations

CANDIDATES = {
    'elasticnet': {'elasticnet': 1.0},
    'base_lgbm': {'base_lgbm': 1.0},
    'elasticnet_lgbm_equal': {'elasticnet': .5, 'base_lgbm': .5},
    'elasticnet_lgbm_lstm_equal': {'elasticnet': 1/3, 'base_lgbm': 1/3, 'lstm': 1/3},
    'original_40_40_20': {'mlp': .4, 'macro': .4, 'lstm': .2},
}

def settle_with_loss50(picked, calendar, prices, entry, end, target_return, requested_n):
    """Keep observed results; mark incomplete holdings at 50% of entry at exit.

    Observed marks are carried through gaps; absent entry marks stay flat until
    exit. The resulting MDD is a scenario, not an observed drawdown.
    """
    observed, wealth = protocol.settle_path(
        picked, calendar, prices, entry, end, target_return, requested_n)
    out = dict(observed)
    out.update(selection_return=observed['return_settled'], selection_mdd=observed['mdd'],
               selection_path_status=observed['status'], assumed_holdings=0,
               selection_basis='observed_path')
    if wealth is not None or len(picked) != requested_n or requested_n <= 0:
        return out, wealth
    dates = calendar[(calendar >= entry) & (calendar <= end)]
    if len(dates) < 2 or dates[0] != entry or dates[-1] != end:
        return out, None
    block = prices.reindex(index=dates, columns=picked.Ticker.tolist()).astype(float)
    block = block.where(np.isfinite(block) & block.gt(0))
    returns = picked[target_return].to_numpy(float)
    endpoints = block.iloc[-1].to_numpy()/block.iloc[0].to_numpy()-1
    unresolved = (~np.isfinite(returns) | block.isna().any().to_numpy()
                  | ~np.isclose(endpoints, returns, atol=1e-6, rtol=1e-5))
    normalized = block.div(block.iloc[0]).ffill().fillna(1.0)
    normalized.iloc[-1, np.flatnonzero(unresolved)] = .5
    wealth = normalized.mean(axis=1)
    out.update(selection_return=float(wealth.iloc[-1]-1),
               selection_mdd=float((wealth/wealth.cummax()-1).min()),
               selection_path_status='settled_path', assumed_holdings=int(unresolved.sum()),
               selection_basis='loss50_terminal_scenario')
    return out, wealth


def summarize_selection(periods, paths):
    scenario = periods.copy()
    scenario['return_settled'] = scenario.selection_return
    scenario['status'] = scenario.selection_path_status
    summary = protocol.summarize_paths(scenario, paths)
    summary['observed_settled_periods'] = summary.strategy.map(
        periods.groupby('strategy').return_settled.count())
    summary['assumed_holdings'] = summary.strategy.map(
        periods.groupby('strategy').assumed_holdings.sum())
    summary['selection_basis'] = np.where(summary.assumed_holdings.gt(0),
                                         'loss50_terminal_scenario', 'observed_path')
    return summary


def select_candidate(summary, metrics, expected_periods):
    """Rank explicitly labelled loss50 scenarios alongside observed paths."""
    ranked = summary.copy()
    ranked['mean_ic'] = ranked.strategy.map(metrics.groupby('strategy').daily_ic.mean())
    ranked['eligible'] = (
        ranked.n_periods.eq(expected_periods) & ranked.status.eq('complete_path')
        & np.isfinite(ranked[['cumulative_return', 'mdd']]).all(axis=1)
    )
    ranked['risk_preference_met'] = ranked.eligible & ranked.mdd.ge(-.15)
    pool = ranked.loc[ranked.risk_preference_met].sort_values(
        ['cumulative_return', 'mdd', 'mean_ic', 'strategy'],
        ascending=[False, False, False, True], na_position='last')
    winner = pool.strategy.iloc[0] if len(pool) else None
    ranked['selected'] = ranked.strategy.eq(winner)
    ranked['selection_status'] = 'exploratory_selected' if winner else 'deferred_no_complete_risk_qualified_candidate'
    ranked['evaluation_role'] = 'exploratory_reused_oos'
    ranked['independent_holdout'] = False
    return ranked, winner

def run_comparison(frame, calendar, sectors, macro, policy):
    candidates = dict(CANDIDATES)
    if p.torch is None:
        raise RuntimeError('Five-candidate common-sample comparison requires PyTorch.')
    labelled, cal, cols = protocol.make_horizon_labels(frame, 120, 1)
    mf = p._macro_by_date(macro, cal)
    for column in p.MACRO_COLS:
        labelled[column] = labelled.Date.map(mf[column]).astype('float32')
    lookup = labelled[['Date', 'Ticker', 'Close']].set_index(['Date', 'Ticker']).Close
    keys = pd.MultiIndex.from_arrays([labelled[cols['end']], labelled.Ticker], names=['Date', 'Ticker'])
    labelled['future_price'] = lookup.reindex(keys).to_numpy()
    del lookup, keys
    features = list(dict.fromkeys(p.MICRO_COLS + p.MACRO_COLS))
    cache = p.StabilityLabelCache(frame, policy)
    bank = p._SequenceBank(frame, cal)
    prices = frame.pivot(index='Date', columns='Ticker', values='Close')
    mapping = p._sector_map(sectors)
    positions = np.arange(cal.searchsorted(pd.Timestamp('2024-07-01')), len(cal)-121, 120)
    if not len(positions):
        raise ValueError('No mature OOS periods')
    metrics, periods, holdings, predictions, losses, audits, paths = [], [], [], [], [], [], {}

    def prepare(day):
        tr, te = p._prepare_control_fold(labelled, cache, day, cols, features)
        tr, te = bank.filter(tr), bank.filter(te)
        tr = tr.loc[np.isfinite(tr[features+[cols['target']]].to_numpy(float)).all(axis=1)].copy()
        # Future labels must never determine the prediction population.
        te = te.loc[np.isfinite(te[features].to_numpy(float)).all(axis=1)].copy()
        te['Sector'] = te.Ticker.map(mapping).fillna('UNKNOWN')
        if len(tr) < 50 or te.empty:
            raise ValueError(f'Insufficient common sample at {day}')
        assert tr[cols['end']].lt(day).all() and tr.label_available_at.lt(day).all()
        return tr, te

    def score(tr, te, day):
        models = {
            'elasticnet': (p.BASE, make_pipeline(RobustScaler(), ElasticNet(alpha=.001, l1_ratio=.5, max_iter=5000))),
            'base_lgbm': (p.BASE, lgb.LGBMRegressor(**p.LGBM_PARAMS, force_col_wise=True)),
            'mlp': (p.BASE, p._final_ensemble_mlp()),
            'macro': (features, lgb.LGBMRegressor(**p.LGBM_PARAMS, force_col_wise=True)),
        }
        scores = pd.DataFrame(index=te.index)
        for name, (inputs, model) in models.items():
            print(f'  {day.date()} {name}', flush=True)
            with threadpool_limits(limits=4):
                scores[name] = p._fit_predict(tr, te, inputs, cols['target'], model)
        print(f'  {day.date()} lstm', flush=True)
        scores['lstm'], info, history = p._fit_final_ensemble_lstm(tr, te, bank, cols['target'])
        assert np.isfinite(scores.to_numpy()).all()
        ranks = scores.rank(pct=True)
        for name, weights in candidates.items():
            scores['candidate_'+name] = sum(ranks[model]*weight for model, weight in weights.items())
        return scores, info, history

    # Freeze weights before evaluating any OOS portfolio. Each annual model is
    # fitted once, against six snapshots, with label-end purging at year start.
    cv_rows, cv_audit = [], []
    for year in (2020, 2021, 2022):
        days = cal[cal >= pd.Timestamp(f'{year}-01-01')][:120:20]
        tr, _ = prepare(days[0])
        tests = []
        for day in days:
            unused, te = prepare(day)
            tests.append(te)
            del unused
        te = pd.concat(tests).sort_values(['Date', 'Ticker']).reset_index(drop=True)
        scores, info, _ = score(tr, te, days[0])
        for day, group in te.groupby('Date'):
            for model in MODELS:
                actual = group[cols['target']]
                predicted = scores.loc[group.index, model]
                cv_rows.append(dict(Date=day, model=model,
                    daily_ic=predicted.corr(actual, method='spearman')))
        cv_audit.append(dict(cv_year=year, fold_start=days[0],
            max_train_label_end=tr[cols['end']].max(),
            max_label_available=tr.label_available_at.max(), **info))
        del tr, te, scores, tests
        gc.collect()
    cv_ic = pd.DataFrame(cv_rows)
    weights = relu_weights(cv_ic.groupby('model').daily_ic.mean())
    final_name = 'cv_locked_relu_4model'
    candidates[final_name] = weights

    for fold, pos in enumerate(positions, 1):
        day, entry, end = cal[pos], cal[pos+1], cal[pos+121]
        print(f'Candidate comparison fold {fold}/{len(positions)}: {day.date()}', flush=True)
        tr, te = prepare(day)
        scores, info, history = score(tr, te, day)
        audits.append(dict(Date=day, train_rows=len(tr), candidates=len(te),
            max_train_label_end=tr[cols['end']].max(), max_label_available=tr.label_available_at.max(), **info))
        losses.extend(dict(Date=day, fold=fold, **item) for item in history)
        predictions.append(pd.concat([te[['Date','Ticker',cols['target'],cols['simple']]], scores], axis=1))
        for name in candidates:
            evaluated = te.copy()
            evaluated['pred'] = scores['candidate_'+name]
            report, _, _ = p.qv.evaluation(evaluated, 'pred', cols['target'])
            metrics.append(dict(Date=day, fold=fold, strategy=name, **report))
            picked = p._choose_top(evaluated, 'pred', 10)
            stat, wealth = settle_with_loss50(picked, cal, prices, entry, end, cols['simple'], 10)
            periods.append(dict(Date=day, EntryDate=entry, ExitDate=end, strategy=name, top_n=10, **stat))
            if wealth is not None:
                paths[(name,10,day)] = wealth
            picked['strategy'], picked['weight'] = name, .1
            holdings.append(picked)
        del tr, te, scores, evaluated, picked
        gc.collect()

    metrics, periods = pd.DataFrame(metrics), pd.DataFrame(periods)
    summary = summarize_selection(periods, paths)
    ranking, winner = select_candidate(summary, metrics, len(positions))
    # Exploratory ranking is retained, but cannot choose the live recommender.
    ranking['exploratory_selected'] = ranking['selected']
    winner = final_name
    ranking['selected'] = ranking.strategy.eq(winner)
    ranking.loc[ranking.selected, 'selection_status'] = 'cv_weights_locked'
    ranking.loc[ranking.selected, 'evaluation_role'] = 'cv_only_weight_selection_reused_oos_evaluation'
    print(ranking[['strategy','settled_periods','cumulative_return','mdd','mean_ic','selected']].to_string(index=False), flush=True)
    recommendation = pd.DataFrame()
    latest = labelled.loc[labelled.eligible_signal, 'Date'].max()
    if winner:
        tr, te = prepare(latest)
        scores, _, _ = score(tr, te, latest)
        for model in MODELS:
            te[model] = scores[model]
            te[model + '_rank_pct'] = scores[model].rank(pct=True)
        te['pred'] = scores['candidate_'+winner]
        recommendation = p._choose_top(te, 'pred', 10).sort_values('pred', ascending=False).copy()
        recommendation['strategy'] = winner
        recommendation['weight'] = .1
        recommendation['signal_date'] = latest
        recommendation['recommendation_status'] = 'unsettled_live_signal'
        recommendation['holding_horizon_trading_days'] = 120
        recommendation['intended_entry'] = 'next_trading_day'
        recommendation = explain_recommendations(recommendation, weights)
    exposure = (recommendation.groupby('Sector', as_index=False).agg(weight=('weight','sum'))
        if len(recommendation) else pd.DataFrame(columns=['Sector','weight']))
    rule = pd.DataFrame([dict(selected_strategy=winner or 'NONE', expected_periods=len(positions),
        rule='Fixed four-model ReLU weights from 2020-2022 purged CV IC only; no OOS performance selection; unresolved holding -50% at exit',
        status='cv_weights_locked',
        evaluation_role='cv_only_weight_selection_reused_oos_evaluation', independent_holdout=False,
        sample='common macro/base features and 20-session sequence; no future-label filtering of candidates',
        costs='before_costs', latest_signal=latest)])
    result = {
        'cv_locked_component_ic': cv_ic,
        'cv_locked_training_audit': pd.DataFrame(cv_audit),
        'cv_locked_weights': pd.DataFrame([dict(model=m, mean_cv_ic=cv_ic.loc[cv_ic.model.eq(m),'daily_ic'].mean(), selected_relu_weight=w) for m,w in weights.items()]),
        'profit_candidate_summary': ranking, 'profit_candidate_selection': rule,
        'profit_candidate_metrics': metrics, 'profit_candidate_periods': periods,
        'profit_candidate_predictions': pd.concat(predictions, ignore_index=True),
        'profit_candidate_audit': pd.DataFrame(audits),
        'profit_candidate_holdings': pd.concat(holdings, ignore_index=True),
        'profit_candidate_weights': pd.DataFrame(candidates).fillna(0).rename_axis('model').reset_index(),
        'profit_final_ensemble_protocol': rule,
        'profit_final_ensemble_summary': ranking.loc[ranking.selected].copy(),
        'profit_final_ensemble_metrics': metrics.loc[metrics.strategy.eq(winner)].copy(),
        'profit_final_ensemble_periods': periods.loc[periods.strategy.eq(winner)].copy(),
        'profit_final_ensemble_holdings': pd.concat(holdings, ignore_index=True).loc[lambda x: x.strategy.eq(winner)],
        'profit_final_ensemble_recommendation': recommendation,
        'profit_final_ensemble_sector_exposure': exposure,
        'profit_final_ensemble_lstm_losses': pd.DataFrame(losses),
    }
    gc.collect()
    return result
