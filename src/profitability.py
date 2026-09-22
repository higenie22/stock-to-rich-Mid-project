from __future__ import annotations

from dataclasses import dataclass
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

try:
    import torch
    from torch import nn
except ImportError:  # reported explicitly by run_lstm_suite
    torch = None
    nn = None

from src import quant_validation as qv
from src import research_protocol as protocol

from .config import ProjectConfig
from .stability import StabilityPolicy


BASE = ["ma_gap_120d", "momentum_120d", "relative_momentum_120d", "rsi_56d"]
DIFF = [
    f"{prefix}_diff_{span}"
    for prefix in ["momentum", "relative_momentum"]
    for span in ["short", "mid", "long"]
]
VOLUME = ["relative_volume_20d", "volume_trend_20_120"]
MACRO_COLS = [
    "Yield_Curve_Spread", "Yield_Curve_Change_20d", "VIX_Level", "VIX_Change_20d",
    "DXY_Ret_60d", "Oil_Ret_60d", "HYG_Ret_60d",
]
INTERACTION_COLS = ["Beta_x_YieldSpread", "Downside_x_VIX", "Sortino_div_DXY"]
MICRO_COLS = BASE + ["beta_120d", "downside_vol_120d", "sortino_120d"]
HORIZON_FEATURES = {
    20: ["ma_gap_20d", "momentum_20d", "relative_momentum_20d", "rsi_14d"],
    60: ["ma_gap_60d", "momentum_60d", "relative_momentum_60d", "rsi_28d"],
    120: BASE,
}
LGBM_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=4,
    min_child_samples=100,
    reg_lambda=5,
    random_state=42,
    n_jobs=1,
    verbosity=-1,
)
FINAL_ENSEMBLE_WEIGHTS = {"mlp": 0.40, "macro_lgbm": 0.40, "lstm": 0.20}
FINAL_ENSEMBLE_EVALUATION_ROLE = "exploratory_reused_oos"


@dataclass
class StabilityLabelCache:
    frame: pd.DataFrame
    policy: StabilityPolicy

    def __post_init__(self) -> None:
        self._parts: dict[pd.Timestamp, pd.DataFrame] = {}

    def get(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        dates = pd.DatetimeIndex(dates).unique()
        missing = [pd.Timestamp(day) for day in dates if pd.Timestamp(day) not in self._parts]
        if missing:
            labels = self.policy.label_dates(self.frame, missing)
            for day, group in labels.groupby("Date"):
                self._parts[pd.Timestamp(day)] = group.copy()
            for day in missing:
                self._parts.setdefault(
                    day,
                    pd.DataFrame(columns=["Date", "Ticker", "risk_rank", "stability_group", "stable"]),
                )
        parts = [self._parts[pd.Timestamp(day)] for day in dates]
        parts = [part for part in parts if not part.empty]
        if not parts:
            return pd.DataFrame(columns=["Date", "Ticker", "risk_rank", "stability_group", "stable"])
        result = pd.concat(parts, ignore_index=True)
        result["Date"] = pd.to_datetime(result.Date)
        return result


def _with_stability(frame: pd.DataFrame, cache: StabilityLabelCache) -> pd.DataFrame:
    labels = cache.get(pd.DatetimeIndex(frame.Date.unique()))
    return frame.merge(labels, on=["Date", "Ticker"], how="inner", validate="one_to_one")


def _sector_map(sectors: pd.DataFrame) -> dict[str, str]:
    return sectors.set_index("Ticker").Sector.to_dict()


def _choose_top(group: pd.DataFrame, score: str, n: int, sector_cap: float = 0.5) -> pd.DataFrame:
    limit = max(1, int(np.floor(n * sector_cap)))
    chosen = []
    counts: dict[str, int] = {}
    for row in group.sort_values(score, ascending=False).itertuples():
        sector = getattr(row, "Sector", "UNKNOWN")
        if counts.get(sector, 0) >= limit:
            continue
        chosen.append(row.Index)
        counts[sector] = counts.get(sector, 0) + 1
        if len(chosen) == n:
            break
    return group.loc[chosen].copy()


def _low_vol(group: pd.DataFrame, n: int, stable_only: bool = False) -> pd.DataFrame:
    source = group.loc[group.stable] if stable_only else group
    return source.sort_values("vol_ann_120d").head(n).copy()


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    model,
) -> np.ndarray:
    ready_train = train.loc[np.isfinite(train[features + [target]].to_numpy(float)).all(axis=1)]
    ready_test = test.loc[np.isfinite(test[features].to_numpy(float)).all(axis=1)]
    if len(ready_train) < 50 or len(ready_test) != len(test):
        raise ValueError("학습 또는 예측 공통 피처 표본이 부족합니다.")
    x_train = np.ascontiguousarray(ready_train[features].to_numpy(dtype=np.float32))
    y_train = np.ascontiguousarray(ready_train[target].to_numpy(dtype=np.float64))
    x_test = np.ascontiguousarray(test[features].to_numpy(dtype=np.float32))
    fitted = clone(model).fit(x_train, y_train)
    prediction = fitted.predict(x_test)
    del fitted, x_train, y_train, x_test
    gc.collect()
    return prediction


def _fit_ranker(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    ready = train.loc[np.isfinite(train[features + [target]].to_numpy(float)).all(axis=1)].copy()
    ready = ready.sort_values(["Date", "Ticker"])
    relevance = (
        (ready.groupby("Date")[target].rank(method="first", pct=True) * 5)
        .apply(np.ceil).clip(1, 5).astype(int) - 1
    )
    model = lgb.LGBMRanker(
        **LGBM_PARAMS, objective="lambdarank", label_gain=[0, 1, 3, 7, 15]
    )
    model.fit(
        np.ascontiguousarray(ready[features].to_numpy(np.float32)),
        relevance.to_numpy(),
        group=ready.groupby("Date", sort=False).size().to_list(),
    )
    prediction = model.predict(np.ascontiguousarray(test[features].to_numpy(np.float32)))
    del model, ready, relevance
    gc.collect()
    return prediction


def run_horizon_suite(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sectors: pd.DataFrame,
    policy: StabilityPolicy,
    config: ProjectConfig,
) -> dict[str, pd.DataFrame]:
    """Horizon, training-scope and portfolio hypotheses with one Stable policy."""
    cache = StabilityLabelCache(frame, policy)
    sector_lookup = _sector_map(sectors)
    price_panel = frame.pivot(index="Date", columns="Ticker", values="Close")
    period_rows, metric_rows, holding_rows, path_map = [], [], [], {}
    for horizon in [20, 60, 120]:
        if "labelled" in locals():
            del labelled, train, test, fit_data, picked, selected
            gc.collect()
        labelled, horizon_calendar, cols = protocol.make_horizon_labels(frame, horizon, 1)
        positions = np.arange(horizon_calendar.searchsorted(pd.Timestamp("2024-07-01")), len(horizon_calendar), horizon)
        positions = positions[positions + 1 + horizon < len(horizon_calendar)]
        for fold, position in enumerate(positions, 1):
            day, entry, end = (
                horizon_calendar[position],
                horizon_calendar[position + 1],
                horizon_calendar[position + 1 + horizon],
            )
            features = HORIZON_FEATURES[horizon]
            train_mask = (
                labelled.eligible_signal
                & labelled.Date.lt(day)
                & labelled[cols["end"]].lt(day)
                & labelled.label_available_at.lt(day)
            )
            train_dates = pd.DatetimeIndex(labelled.loc[train_mask, "Date"].unique()).sort_values()[::5]
            needed = list(dict.fromkeys(
                ["Date", "Ticker", "Close", "eligible_signal", "label_available_at",
                 cols["target"], cols["simple"], cols["end"], "vol_ann_120d", *features]
            ))
            train = labelled.loc[train_mask & labelled.Date.isin(train_dates), needed].copy()
            train = _with_stability(train, cache)
            test = _with_stability(
                labelled.loc[labelled.eligible_signal & labelled.Date.eq(day), needed].copy(), cache
            )
            ready = np.isfinite(test[features].to_numpy(float)).all(axis=1)
            test = test.loc[ready].copy()
            test["Sector"] = test.Ticker.map(sector_lookup).fillna("UNKNOWN")
            model = lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True)
            for scope in ["all_train", "stable_train"]:
                fit_data = train if scope == "all_train" else train.loc[train.stable]
                test[f"pred_{scope}"] = _fit_predict(fit_data, test, features, cols["target"], model)
                report, _, _ = qv.evaluation(test.loc[test.stable], f"pred_{scope}", cols["target"])
                metric_rows.append(
                    dict(
                        horizon=horizon,
                        fold=fold,
                        Date=day,
                        scope=scope,
                        train_rows=len(fit_data),
                        max_train_label_end=fit_data[cols["end"]].max(),
                        max_label_available=fit_data.label_available_at.max(),
                        **report,
                    )
                )
            selections = {
                "lowvol": _low_vol(test, 10, False),
                "stable_lowvol": _low_vol(test, 10, True),
                "all_train": _choose_top(test.loc[test.stable], "pred_all_train", 10),
                "stable_train": _choose_top(test.loc[test.stable], "pred_stable_train", 10),
            }
            for name, picked in selections.items():
                metrics, wealth = protocol.settle_path(
                    picked, horizon_calendar, price_panel, entry, end, cols["simple"], 10
                )
                period_rows.append(
                    dict(
                        horizon=horizon,
                        fold=fold,
                        Date=day,
                        EntryDate=entry,
                        ExitDate=end,
                        strategy=name,
                        top_n=10,
                        **metrics,
                    )
                )
                if wealth is not None:
                    path_map[(horizon, name, 10, day)] = wealth
                selected = picked.copy()
                selected["horizon"] = horizon
                selected["strategy"] = name
                selected["top_n"] = 10
                holding_rows.append(selected)
    periods = pd.DataFrame(period_rows)
    summaries = []
    for horizon, group in periods.groupby("horizon"):
        paths = {
            (strategy, n, day): value
            for (h, strategy, n, day), value in path_map.items()
            if h == horizon
        }
        summary = protocol.summarize_paths(group, paths)
        summary["horizon"] = horizon
        summaries.append(summary)
    return {
        "profit_horizon_metrics": pd.DataFrame(metric_rows),
        "profit_horizon_periods": periods,
        "profit_horizon_summary": pd.concat(summaries, ignore_index=True),
        "profit_holdings": pd.concat(holding_rows, ignore_index=True),
    }


def _prepare_control_fold(
    labelled: pd.DataFrame,
    cache: StabilityLabelCache,
    day: pd.Timestamp,
    cols: dict,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (
        labelled.eligible_signal
        & labelled.Date.lt(day)
        & labelled[cols["end"]].lt(day)
        & labelled.label_available_at.lt(day)
    )
    dates = pd.DatetimeIndex(labelled.loc[train_mask, "Date"].unique()).sort_values()[::5]
    needed = list(dict.fromkeys(
        ["Date", "Ticker", "Close", "eligible_signal", "label_available_at",
         cols["entry"], cols["end"], cols["target"], cols["simple"], "future_price",
         "momentum_252d", "beta_120d", *features]
    ))
    train = _with_stability(
        labelled.loc[train_mask & labelled.Date.isin(dates), needed].copy(), cache
    )
    test = _with_stability(
        labelled.loc[labelled.eligible_signal & labelled.Date.eq(day), needed].copy(), cache
    )
    common = list(dict.fromkeys(features))
    train = train.loc[train.stable & np.isfinite(train[common].to_numpy(float)).all(axis=1)].copy()
    test = test.loc[test.stable & np.isfinite(test[common].to_numpy(float)).all(axis=1)].copy()
    return train, test


def run_controlled_models(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sectors: pd.DataFrame,
    policy: StabilityPolicy,
) -> dict[str, pd.DataFrame]:
    """Target, feature and model ablations from the validation report."""
    labelled, horizon_calendar, cols = protocol.make_horizon_labels(frame, 120, 1)
    lookup = labelled.set_index(["Date", "Ticker"]).Close
    exit_keys = pd.MultiIndex.from_arrays([labelled[cols["end"]], labelled.Ticker], names=["Date", "Ticker"])
    labelled["future_price"] = lookup.reindex(exit_keys).to_numpy()
    cache = StabilityLabelCache(frame, policy)
    sector_lookup = _sector_map(sectors)
    feature_union = list(dict.fromkeys(BASE + DIFF + VOLUME))
    positions = np.arange(horizon_calendar.searchsorted(pd.Timestamp("2024-07-01")), len(horizon_calendar), 120)
    positions = positions[positions + 121 < len(horizon_calendar)]
    cases = {
        "base_excess_lgbm": (BASE, cols["target"], lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True), None),
        "absolute_target": (BASE, cols["simple"], lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True), None),
        "future_price_target": (BASE, "future_price", lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True), None),
        "difference_features": (BASE + DIFF, cols["target"], lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True), None),
        "volume_features": (BASE + VOLUME, cols["target"], lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True), None),
        "rolling_3y": (BASE, cols["target"], lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True), "rolling"),
        "ranker": (BASE, cols["target"], None, "ranker"),
        "elasticnet": (
            BASE,
            cols["target"],
            make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),
            None,
        ),
        "mlp": (
            BASE,
            cols["target"],
            make_pipeline(
                RobustScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16), max_iter=100, shuffle=False,
                    random_state=42, alpha=0.1, batch_size=1024,
                ),
            ),
            None,
        ),
    }
    # H3-3: direction is fixed using past-only 2020~2022 snapshots.
    direction_rows = []
    for year in [2020, 2021, 2022]:
        cv_days = horizon_calendar[horizon_calendar >= pd.Timestamp(f"{year}-01-01")][:120:20]
        for day in cv_days:
            cv_train, cv_test = _prepare_control_fold(labelled, cache, day, cols, feature_union)
            cv_prediction = _fit_predict(
                cv_train, cv_test, BASE, cols["target"],
                lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True),
            )
            cv_eval = cv_test.copy()
            cv_eval["pred"] = cv_prediction
            cv_report, _, _ = qv.evaluation(cv_eval, "pred", cols["target"])
            direction_rows.append({"year": year, "Date": day, **cv_report})
            del cv_train, cv_test, cv_eval, cv_prediction
            gc.collect()
    direction_cv = pd.DataFrame(direction_rows)
    direction = -1 if len(direction_cv) and direction_cv.daily_ic.mean() < 0 else 1

    price_panel = frame.pivot(index="Date", columns="Ticker", values="Close")
    metric_rows, prediction_rows, period_rows, holding_rows, paths = [], [], [], [], {}
    for fold, position in enumerate(positions, 1):
        day = horizon_calendar[position]
        train, test = _prepare_control_fold(labelled, cache, day, cols, feature_union)
        test["Sector"] = test.Ticker.map(sector_lookup).fillna("UNKNOWN")
        base_prediction = None
        for name, (features, target, model, mode) in cases.items():
            fit_data = train.loc[train.Date.ge(day - pd.DateOffset(years=3))] if mode == "rolling" else train
            prediction = (
                _fit_ranker(fit_data, test, features, target)
                if mode == "ranker"
                else _fit_predict(fit_data, test, features, target, model)
            )
            evaluated = test.copy()
            evaluated["pred"] = prediction
            report, _, _ = qv.evaluation(evaluated, "pred", cols["target"])
            metric_rows.append(
                dict(
                    fold=fold,
                    Date=day,
                    case=name,
                    fit_rows=len(fit_data),
                    train_start=fit_data.Date.min(),
                    train_end=fit_data.Date.max(),
                    **report,
                )
            )
            if name == "base_excess_lgbm":
                base_prediction = prediction.copy()
                prediction_rows.append(evaluated)
            picked = _choose_top(evaluated, "pred", 10)
            entry, end = horizon_calendar[position + 1], horizon_calendar[position + 121]
            settled, wealth = protocol.settle_path(
                picked, horizon_calendar, price_panel, entry, end, cols["simple"], 10
            )
            period_rows.append(
                dict(Date=day, EntryDate=entry, ExitDate=end, case=name, top_n=10, **settled)
            )
            if wealth is not None:
                paths[(name, 10, day)] = wealth
            picked = picked.copy()
            picked["case"] = name
            holding_rows.append(picked)
        if base_prediction is not None:
            evaluated = test.copy()
            evaluated["pred"] = base_prediction * direction
            report, _, _ = qv.evaluation(evaluated, "pred", cols["target"])
            metric_rows.append(
                dict(fold=fold, Date=day, case="cv_direction", fit_rows=len(train),
                     train_start=train.Date.min(), train_end=train.Date.max(), **report)
            )
            picked = _choose_top(evaluated, "pred", 10)
            entry, end = horizon_calendar[position + 1], horizon_calendar[position + 121]
            settled, wealth = protocol.settle_path(
                picked, horizon_calendar, price_panel, entry, end, cols["simple"], 10
            )
            period_rows.append(
                dict(Date=day, EntryDate=entry, ExitDate=end, case="cv_direction", top_n=10, **settled)
            )
            if wealth is not None:
                paths[("cv_direction", 10, day)] = wealth
            picked = picked.copy(); picked["case"] = "cv_direction"; holding_rows.append(picked)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    factor = pd.DataFrame()
    if len(predictions):
        factor, _ = qv.incremental_factor_ic(predictions, cols["target"], sectors=None)
    periods = pd.DataFrame(period_rows).rename(columns={"case": "strategy"})
    summary = protocol.summarize_paths(periods, paths) if len(periods) else pd.DataFrame()
    return {
        "profit_controlled_models": metrics,
        "profit_direction_cv": direction_cv.assign(selected_direction=direction),
        "profit_controlled_periods": periods,
        "profit_controlled_summary": summary,
        "profit_controlled_holdings": pd.concat(holding_rows, ignore_index=True),
        "profit_base_predictions": predictions,
        "profit_factor_robustness": factor,
    }


def _macro_by_date(raw: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    source = raw.rename(
        columns={"^TNX": "US10Y", "^IRX": "US3M", "^VIX": "VIX", "DX-Y.NYB": "DXY", "CL=F": "Oil", "HYG": "HYG"}
    ).reindex(calendar).ffill(limit=3)
    result = pd.DataFrame(index=calendar)
    result["Yield_Curve_Spread"] = source.US10Y - source.US3M
    result["Yield_Curve_Change_20d"] = result.Yield_Curve_Spread.diff(20)
    result["VIX_Level"] = source.VIX
    result["VIX_Change_20d"] = source.VIX / source.VIX.shift(20) - 1
    for name in ["DXY", "Oil", "HYG"]:
        positive = source[name].where(source[name] > 0)
        result[f"{name}_Ret_60d"] = np.log(positive / positive.shift(60))
    return result.shift(1).rename_axis("Date")


def run_macro_suite(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sectors: pd.DataFrame,
    macro: pd.DataFrame,
    policy: StabilityPolicy,
) -> dict[str, pd.DataFrame]:
    labelled, horizon_calendar, cols = protocol.make_horizon_labels(frame, 120, 1)
    macro_features = _macro_by_date(macro, horizon_calendar)
    for column in MACRO_COLS:
        labelled[column] = labelled.Date.map(macro_features[column]).astype("float32")
    labelled["Beta_x_YieldSpread"] = labelled.beta_120d * labelled.Yield_Curve_Spread
    labelled["Downside_x_VIX"] = labelled.downside_vol_120d * labelled.VIX_Level / 100
    labelled["Sortino_div_DXY"] = labelled.sortino_120d / (np.exp(labelled.DXY_Ret_60d) + 1e-5)
    lookup = labelled[["Date", "Ticker", "Close"]].set_index(["Date", "Ticker"]).Close
    keys = pd.MultiIndex.from_arrays([labelled[cols["end"]], labelled.Ticker], names=["Date", "Ticker"])
    labelled["future_price"] = lookup.reindex(keys).to_numpy()
    cache = StabilityLabelCache(frame, policy)
    sector_lookup = _sector_map(sectors)
    configs = {
        "micro_base": MICRO_COLS,
        "macro_added": MICRO_COLS + MACRO_COLS,
        "interactions_added": MICRO_COLS + INTERACTION_COLS,
    }
    common = list(dict.fromkeys(sum(configs.values(), [])))
    positions = np.arange(horizon_calendar.searchsorted(pd.Timestamp("2024-07-01")), len(horizon_calendar), 120)
    positions = positions[positions + 121 < len(horizon_calendar)]
    rows = []
    for fold, position in enumerate(positions, 1):
        day = horizon_calendar[position]
        train, test = _prepare_control_fold(labelled, cache, day, cols, common)
        test["Sector"] = test.Ticker.map(sector_lookup).fillna("UNKNOWN")
        for name, features in configs.items():
            prediction = _fit_predict(
                train, test, features, cols["target"],
                lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True),
            )
            evaluated = test.copy(); evaluated["pred"] = prediction
            report, _, _ = qv.evaluation(evaluated, "pred", cols["target"])
            rows.append(dict(fold=fold, Date=day, case=name, fit_rows=len(train), **report))
        del train, test
        gc.collect()
    return {
        "profit_macro_metrics": pd.DataFrame(rows),
        "profit_macro_features": macro_features.reset_index(),
    }


class _SequenceBank:
    def __init__(self, frame: pd.DataFrame, calendar: pd.DatetimeIndex, lookback: int = 20):
        self.calendar = calendar
        self.lookback = lookback
        self.tickers = pd.Index(sorted(frame.Ticker.unique()))
        arrays = []
        for feature in BASE:
            arrays.append(
                frame.pivot(index="Date", columns="Ticker", values=feature)
                .reindex(index=calendar, columns=self.tickers)
                .to_numpy(dtype=np.float32)
            )
        self.data = np.stack(arrays, axis=-1)
        del arrays
        valid = np.isfinite(self.data).all(axis=-1)
        self.ready = (
            pd.DataFrame(valid, index=calendar)
            .rolling(lookback, min_periods=lookback).sum().eq(lookback).to_numpy()
        )

    def _indices(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.calendar.get_indexer(frame.Date), self.tickers.get_indexer(frame.Ticker)

    def filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        date_index, ticker_index = self._indices(frame)
        mask = (date_index >= self.lookback - 1) & (ticker_index >= 0)
        mask &= self.ready[np.maximum(date_index, 0), np.maximum(ticker_index, 0)]
        return frame.loc[mask].copy()

    def take(self, frame: pd.DataFrame) -> np.ndarray:
        date_index, ticker_index = self._indices(frame)
        offsets = np.arange(self.lookback - 1, -1, -1)
        return self.data[date_index[:, None] - offsets, ticker_index[:, None], :]


if nn is not None:
    class _ReportLSTM(nn.Module):
        def __init__(self, feature_count: int):
            super().__init__()
            self.rnn = nn.LSTM(feature_count, 32, 2, batch_first=True, dropout=0.3)
            self.head = nn.Sequential(
                nn.LayerNorm(32), nn.Dropout(0.3), nn.Linear(32, 32), nn.GELU(),
                nn.Dropout(0.15), nn.Linear(32, 1),
            )

        def forward(self, values):
            return self.head(self.rnn(values)[0][:, -1]).squeeze(-1)


def _train_sequence_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    bank: _SequenceBank,
    target: str,
    epochs: int,
    early_stop: bool,
) -> tuple[np.ndarray, int, list[dict]]:
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(1)
    x_train = bank.take(train)
    scaler = RobustScaler().fit(x_train.reshape(-1, len(BASE)))
    x_train = torch.tensor(
        scaler.transform(x_train.reshape(-1, len(BASE))).reshape(x_train.shape), dtype=torch.float32
    )
    mean = train[target].mean(); std = max(train[target].std(), 1e-6)
    y_train = torch.tensor(((train[target] - mean) / std).to_numpy(), dtype=torch.float32)
    x_valid_np = bank.take(validation)
    x_valid = torch.tensor(
        scaler.transform(x_valid_np.reshape(-1, len(BASE))).reshape(x_valid_np.shape), dtype=torch.float32
    )
    y_valid = torch.tensor(((validation[target] - mean) / std).to_numpy(), dtype=torch.float32)
    model = _ReportLSTM(len(BASE))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=1024, shuffle=True
    )
    best, best_epoch, stale, history = np.inf, 1, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimiser.zero_grad(); loss = loss_function(model(batch_x), batch_y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimiser.step()
        if early_stop:
            model.eval()
            with torch.no_grad():
                loss_value = float(loss_function(model(x_valid), y_valid))
            history.append({"epoch": epoch, "validation_loss": loss_value})
            if loss_value < best - 1e-5:
                best, best_epoch, stale = loss_value, epoch, 0
            else:
                stale += 1
            if stale >= 3:
                break
    model.eval()
    with torch.no_grad():
        prediction = model(x_valid).numpy() * std + mean
    del model, optimiser, loader, x_train, y_train, x_valid, y_valid, x_valid_np
    gc.collect()
    return prediction, best_epoch, history


def run_lstm_suite(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sectors: pd.DataFrame,
    policy: StabilityPolicy,
) -> dict[str, pd.DataFrame]:
    if torch is None:
        return {"profit_lstm_metrics": pd.DataFrame([{"status": "torch_not_installed"}])}
    labelled, horizon_calendar, cols = protocol.make_horizon_labels(frame, 120, 1)
    lookup = labelled[["Date", "Ticker", "Close"]].set_index(["Date", "Ticker"]).Close
    keys = pd.MultiIndex.from_arrays([labelled[cols["end"]], labelled.Ticker], names=["Date", "Ticker"])
    labelled["future_price"] = lookup.reindex(keys).to_numpy()
    bank = _SequenceBank(frame, horizon_calendar)
    cache = StabilityLabelCache(frame, policy)
    positions = np.arange(horizon_calendar.searchsorted(pd.Timestamp("2024-07-01")), len(horizon_calendar), 120)
    positions = positions[positions + 121 < len(horizon_calendar)]
    rows, loss_rows = [], []
    for fold, position in enumerate(positions, 1):
        day = horizon_calendar[position]
        train, test = _prepare_control_fold(labelled, cache, day, cols, BASE)
        train, test = bank.filter(train), bank.filter(test)
        train = train.loc[np.isfinite(train[cols["target"]])].copy()
        validation_start = train.Date.max() - pd.DateOffset(years=1)
        inner_train = train.loc[
            train[cols["end"]].lt(validation_start)
            & train.label_available_at.lt(validation_start)
        ]
        inner_valid = train.loc[train.Date.ge(validation_start)]
        if min(len(inner_train), len(inner_valid), len(test)) == 0:
            continue
        _, selected_epochs, history = _train_sequence_model(
            inner_train, inner_valid, bank, cols["target"], 12, True
        )
        prediction, _, _ = _train_sequence_model(
            train, test, bank, cols["target"], selected_epochs, False
        )
        evaluated = test.copy(); evaluated["pred"] = prediction
        report, _, _ = qv.evaluation(evaluated, "pred", cols["target"])
        rows.append(dict(fold=fold, Date=day, case="lstm", selected_epochs=selected_epochs, **report))
        for item in history:
            loss_rows.append({"fold": fold, "Date": day, **item})
        baseline = _fit_predict(
            train, test, BASE, cols["target"],
            lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True),
        )
        evaluated["pred"] = baseline
        report, _, _ = qv.evaluation(evaluated, "pred", cols["target"])
        rows.append(dict(fold=fold, Date=day, case="same_endpoint_lgbm", selected_epochs=np.nan, **report))
        del train, test, inner_train, inner_valid, evaluated
        gc.collect()
    del bank, labelled
    gc.collect()
    return {
        "profit_lstm_metrics": pd.DataFrame(rows),
        "profit_lstm_losses": pd.DataFrame(loss_rows),
    }


def _final_ensemble_mlp() -> object:
    """The MLP(32, 16) alpha driver specified in the source notebook."""
    return make_pipeline(
        RobustScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=100,
            early_stopping=False,
            shuffle=False,
            random_state=42,
            alpha=0.1,
            batch_size=1024,
            learning_rate_init=0.001,
        ),
    )


def _fit_final_ensemble_lstm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bank: _SequenceBank,
    target: str,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    """Select LSTM epochs on a past-only inner split, then refit for scoring."""
    train = train.loc[np.isfinite(train[target])].copy()
    validation_start = train.Date.max() - pd.DateOffset(years=1)
    inner_train = train.loc[
        train[target].notna()
        & train.Date.lt(validation_start)
        & train.label_available_at.lt(validation_start)
        & train[target].notna()
    ]
    inner_valid = train.loc[train.Date.ge(validation_start) & train[target].notna()]
    if min(len(inner_train), len(inner_valid), len(test)) == 0:
        raise ValueError("최종 앙상블 LSTM의 과거 학습·검증·예측 표본이 부족합니다.")
    _, epochs, history = _train_sequence_model(
        inner_train, inner_valid, bank, target, 12, True
    )
    prediction, _, _ = _train_sequence_model(train, test, bank, target, epochs, False)
    info: dict[str, object] = {
        "selected_epochs": epochs,
        "inner_train_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
        "inner_validation_start": validation_start,
        "inner_max_label_end": inner_train["target_end_120d_next_close"].max(),
    }
    del inner_train, inner_valid
    gc.collect()
    return prediction, info, history


def _rank_blend(evaluated: pd.DataFrame, mlp: np.ndarray, macro: np.ndarray, lstm: np.ndarray) -> pd.DataFrame:
    """Notebook's 40/40/20 cross-sectional percentile-rank blend."""
    result = evaluated.copy()
    result["rank_mlp"] = pd.Series(mlp, index=result.index).rank(pct=True)
    result["rank_macro"] = pd.Series(macro, index=result.index).rank(pct=True)
    result["rank_lstm"] = pd.Series(lstm, index=result.index).rank(pct=True)
    result["pred"] = (
        FINAL_ENSEMBLE_WEIGHTS["mlp"] * result["rank_mlp"]
        + FINAL_ENSEMBLE_WEIGHTS["macro_lgbm"] * result["rank_macro"]
        + FINAL_ENSEMBLE_WEIGHTS["lstm"] * result["rank_lstm"]
    )
    return result


def run_original_final_ensemble(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sectors: pd.DataFrame,
    macro: pd.DataFrame,
    policy: StabilityPolicy,
) -> dict[str, pd.DataFrame]:
    """Run the final MLP/Macro-LGBM/LSTM ensemble added to the source notebook.

    The source weights, 120-day horizon, stable-only training, Top-10 selection
    and 50% sector cap are preserved.  Stable labels intentionally come from
    this submission's single canonical policy rather than the notebook's local
    K-Means reimplementation.
    """
    empty = pd.DataFrame()
    if torch is None:
        return {
            "profit_final_ensemble_metrics": pd.DataFrame([{"status": "torch_not_installed"}]),
            "profit_final_ensemble_periods": empty,
            "profit_final_ensemble_summary": empty,
            "profit_final_ensemble_holdings": empty,
            "profit_final_ensemble_recommendation": empty,
            "profit_final_ensemble_sector_exposure": empty,
            "profit_final_ensemble_lstm_losses": empty,
            "profit_final_ensemble_protocol": pd.DataFrame(
                [{"evaluation_role": FINAL_ENSEMBLE_EVALUATION_ROLE, "status": "torch_not_installed"}]
            ),
        }

    labelled, horizon_calendar, cols = protocol.make_horizon_labels(frame, 120, 1)
    macro_features = _macro_by_date(macro, horizon_calendar)
    for column in MACRO_COLS:
        labelled[column] = labelled.Date.map(macro_features[column]).astype("float32")
    close_lookup = labelled.set_index(["Date", "Ticker"]).Close
    exit_keys = pd.MultiIndex.from_arrays(
        [labelled[cols["end"]], labelled.Ticker], names=["Date", "Ticker"]
    )
    # Required by the shared fold-preparation contract; never used as a feature.
    labelled["future_price"] = close_lookup.reindex(exit_keys).to_numpy()
    ensemble_features = list(dict.fromkeys(MICRO_COLS + MACRO_COLS))
    cache = StabilityLabelCache(frame, policy)
    sector_lookup = _sector_map(sectors)
    bank = _SequenceBank(frame, horizon_calendar)
    price_panel = frame.pivot(index="Date", columns="Ticker", values="Close")
    positions = np.arange(
        horizon_calendar.searchsorted(pd.Timestamp("2024-07-01")), len(horizon_calendar), 120
    )
    positions = positions[positions + 121 < len(horizon_calendar)]
    metric_rows, period_rows, holding_rows, loss_rows, paths = [], [], [], [], {}

    for fold, position in enumerate(positions, 1):
        day = horizon_calendar[position]
        train, test = _prepare_control_fold(labelled, cache, day, cols, ensemble_features)
        train, test = bank.filter(train), bank.filter(test)
        train = train.loc[np.isfinite(train[ensemble_features + [cols["target"]]].to_numpy(float)).all(axis=1)].copy()
        test = test.loc[np.isfinite(test[ensemble_features].to_numpy(float)).all(axis=1)].copy()
        test["Sector"] = test.Ticker.map(sector_lookup).fillna("UNKNOWN")
        if min(len(train), len(test)) == 0:
            metric_rows.append({"fold": fold, "Date": day, "status": "insufficient_candidates"})
            continue

        pred_mlp = _fit_predict(train, test, BASE, cols["target"], _final_ensemble_mlp())
        pred_macro = _fit_predict(
            train, test, ensemble_features, cols["target"],
            lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True),
        )
        pred_lstm, lstm_info, history = _fit_final_ensemble_lstm(train, test, bank, cols["target"])
        evaluated = _rank_blend(test, pred_mlp, pred_macro, pred_lstm)
        report, _, _ = qv.evaluation(evaluated, "pred", cols["target"])
        metric_rows.append(
            {
                "fold": fold,
                "Date": day,
                "strategy": "final_ensemble",
                "evaluation_role": FINAL_ENSEMBLE_EVALUATION_ROLE,
                "independent_holdout": False,
                "fit_rows": len(train),
                **lstm_info,
                **report,
            }
        )
        for item in history:
            loss_rows.append({"fold": fold, "Date": day, **item})
        picked = _choose_top(evaluated, "pred", 10)
        entry, end = horizon_calendar[position + 1], horizon_calendar[position + 121]
        settled, wealth = protocol.settle_path(
            picked, horizon_calendar, price_panel, entry, end, cols["simple"], 10
        )
        period_rows.append(
            {
                "Date": day,
                "EntryDate": entry,
                "ExitDate": end,
                "strategy": "final_ensemble",
                "evaluation_role": FINAL_ENSEMBLE_EVALUATION_ROLE,
                "top_n": 10,
                **settled,
            }
        )
        if wealth is not None:
            paths[("final_ensemble", 10, day)] = wealth
        picked["strategy"] = "final_ensemble"
        picked["weight"] = 1 / 10
        holding_rows.append(picked)
        del train, test, evaluated, pred_mlp, pred_macro, pred_lstm, picked
        gc.collect()

    latest_day = pd.DatetimeIndex(
        labelled.loc[labelled.eligible_signal, "Date"].unique()
    ).max()
    train, candidates = _prepare_control_fold(labelled, cache, latest_day, cols, ensemble_features)
    train, candidates = bank.filter(train), bank.filter(candidates)
    train = train.loc[np.isfinite(train[ensemble_features + [cols["target"]]].to_numpy(float)).all(axis=1)].copy()
    candidates = candidates.loc[np.isfinite(candidates[ensemble_features].to_numpy(float)).all(axis=1)].copy()
    candidates["Sector"] = candidates.Ticker.map(sector_lookup).fillna("UNKNOWN")
    if min(len(train), len(candidates)):
        final_mlp = _fit_predict(train, candidates, BASE, cols["target"], _final_ensemble_mlp())
        final_macro = _fit_predict(
            train, candidates, ensemble_features, cols["target"],
            lgb.LGBMRegressor(**LGBM_PARAMS, force_col_wise=True),
        )
        final_lstm, _, _ = _fit_final_ensemble_lstm(train, candidates, bank, cols["target"])
        recommendation = _choose_top(
            _rank_blend(candidates, final_mlp, final_macro, final_lstm), "pred", 10
        ).sort_values("pred", ascending=False).copy()
        recommendation["weight"] = 1 / 10
        recommendation["signal_date"] = latest_day
        recommendation["recommendation_status"] = "unsettled_live_signal"
        recommendation["intended_entry"] = "next_trading_day"
        recommendation["holding_horizon_trading_days"] = 120
    else:
        recommendation = pd.DataFrame()
    sector_exposure = (
        recommendation.groupby("Sector", as_index=False).agg(weight=("weight", "sum"), holdings=("Ticker", "size"))
        if len(recommendation) else pd.DataFrame(columns=["Sector", "weight", "holdings"])
    )
    periods = pd.DataFrame(period_rows)
    summary = protocol.summarize_paths(periods, paths) if len(periods) else pd.DataFrame()
    if len(summary):
        summary["evaluation_role"] = FINAL_ENSEMBLE_EVALUATION_ROLE
        summary["independent_holdout"] = False
    protocol_table = pd.DataFrame(
        [
            {
                "horizon_trading_days": 120,
                "entry_lag_trading_days": 1,
                "training_scope": "stable_only",
                "blend_method": "cross_sectional_percentile_rank",
                "mlp_weight": FINAL_ENSEMBLE_WEIGHTS["mlp"],
                "macro_lgbm_weight": FINAL_ENSEMBLE_WEIGHTS["macro_lgbm"],
                "lstm_weight": FINAL_ENSEMBLE_WEIGHTS["lstm"],
                "top_n": 10,
                "sector_cap": 0.50,
                "evaluation_role": FINAL_ENSEMBLE_EVALUATION_ROLE,
                "independent_holdout": False,
                "retrospective_current_selection_test": "excluded",
                "next_confirmatory_period": "requires_data_after_2026-06-30",
            }
        ]
    )
    del labelled, bank, train, candidates
    gc.collect()
    return {
        "profit_final_ensemble_metrics": pd.DataFrame(metric_rows),
        "profit_final_ensemble_periods": periods,
        "profit_final_ensemble_summary": summary,
        "profit_final_ensemble_holdings": pd.concat(holding_rows, ignore_index=True) if holding_rows else pd.DataFrame(),
        "profit_final_ensemble_recommendation": recommendation,
        "profit_final_ensemble_sector_exposure": sector_exposure,
        "profit_final_ensemble_lstm_losses": pd.DataFrame(loss_rows),
        "profit_final_ensemble_protocol": protocol_table,
    }


def run_final_ensemble(frame, calendar, sectors, macro, policy):
    from .candidate_comparison import run_comparison
    return run_comparison(frame, calendar, sectors, macro, policy)


def concentration_table(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if holdings.empty:
        return pd.DataFrame()
    for (horizon, day, strategy), group in holdings.groupby(["horizon", "Date", "strategy"]):
        if len(group) == 0:
            continue
        sector_share = group.groupby("Sector").size().max() / len(group) if "Sector" in group else np.nan
        rows.append(
            dict(
                horizon=horizon,
                Date=day,
                strategy=strategy,
                holdings=len(group),
                max_sector_weight=sector_share,
                unique_sectors=group.Sector.nunique() if "Sector" in group else np.nan,
            )
        )
    return pd.DataFrame(rows)


def run_profitability_suite(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sectors: pd.DataFrame,
    macro: pd.DataFrame,
    policy: StabilityPolicy,
    config: ProjectConfig,
) -> dict[str, pd.DataFrame]:
    tables = run_horizon_suite(frame, calendar, sectors, policy, config)
    gc.collect()
    tables.update(run_controlled_models(frame, calendar, sectors, policy))
    gc.collect()
    tables.update(run_macro_suite(frame, calendar, sectors, macro, policy))
    gc.collect()
    tables.update(run_lstm_suite(frame, calendar, sectors, policy))
    gc.collect()
    tables.update(run_final_ensemble(frame, calendar, sectors, macro, policy))
    gc.collect()
    tables["profit_concentration"] = concentration_table(tables["profit_holdings"])
    return tables
