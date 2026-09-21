from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig


@dataclass(frozen=True)
class StabilityPolicy:
    """Canonical Stable definition shared by every experiment.

    It intentionally follows the interpretable v14 policy while consuming the
    validation-report data pipeline: date-wise K=3 on annualized volatility and
    absolute beta, both at the 120-session lookback.
    """

    lookback: int = 120
    k: int = 3
    random_state: int = 42

    @property
    def volatility_column(self) -> str:
        return f"vol_ann_{self.lookback}d"

    @property
    def beta_column(self) -> str:
        return f"beta_{self.lookback}d"

    def label_snapshot(self, snapshot: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        k = self.k if k is None else k
        cols = [self.volatility_column, self.beta_column]
        ready = snapshot.dropna(subset=cols).copy()
        ready = ready.loc[np.isfinite(ready[cols].to_numpy(float)).all(axis=1)]
        if len(ready) < max(k * 4, 12):
            return pd.DataFrame(columns=["Date", "Ticker", "risk_rank", "stability_group", "stable"])
        ready["abs_beta"] = ready[self.beta_column].abs()
        x = ready[[self.volatility_column, "abs_beta"]]
        scaled = StandardScaler().fit_transform(x)
        model = KMeans(n_clusters=k, n_init=20, random_state=self.random_state)
        raw = model.fit_predict(scaled)
        centers = pd.DataFrame(model.cluster_centers_, columns=["z_volatility", "z_abs_beta"])
        order = centers.mean(axis=1).sort_values().index.tolist()
        rank_map = {cluster: rank + 1 for rank, cluster in enumerate(order)}
        ready["risk_rank"] = pd.Series(raw, index=ready.index).map(rank_map).astype(int)
        if k == 3:
            ready["stability_group"] = ready.risk_rank.map({1: "Stable", 2: "Middle", 3: "Risky"})
        else:
            ready["stability_group"] = ready.risk_rank.map(lambda value: f"RiskRank{value}")
        ready["stable"] = ready.risk_rank.eq(1)
        return ready[["Date", "Ticker", "risk_rank", "stability_group", "stable"]]

    def label_dates(self, frame: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        wanted = pd.DatetimeIndex(dates).unique()
        subset = frame.loc[frame.Date.isin(wanted) & frame.eligible_signal].copy()
        parts = [self.label_snapshot(group) for _, group in subset.groupby("Date", sort=True)]
        parts = [part for part in parts if not part.empty]
        if not parts:
            return pd.DataFrame(columns=["Date", "Ticker", "risk_rank", "stability_group", "stable"])
        result = pd.concat(parts, ignore_index=True)
        result["Date"] = pd.to_datetime(result.Date)
        return result


def evaluation_dates(calendar: pd.DatetimeIndex, start: str = "2016-01-01", step: int = 120) -> pd.DatetimeIndex:
    first = calendar.searchsorted(pd.Timestamp(start))
    return calendar[np.arange(first, len(calendar), step)]


def _future_windows(values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    future_vol = np.full(n, np.nan)
    future_mdd = np.full(n, np.nan)
    if n <= horizon + 1:
        return future_vol, future_mdd
    # Signal t -> entry t+1 -> exit t+1+h. This matches the profitability contract.
    windows = np.lib.stride_tricks.sliding_window_view(values[1:], horizon + 1)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    returns = windows[:, 1:] / windows[:, :-1] - 1
    peaks = np.maximum.accumulate(windows, axis=1)
    future_vol[: len(windows)] = np.where(valid, np.std(returns, axis=1, ddof=1) * np.sqrt(252), np.nan)
    future_mdd[: len(windows)] = np.where(valid, np.min(windows / peaks - 1, axis=1), np.nan)
    return future_vol, future_mdd


def forward_outcomes(frame: pd.DataFrame, horizon: int = 120) -> pd.DataFrame:
    parts = []
    for ticker, group in frame[["Date", "Ticker", "Close"]].groupby("Ticker", sort=False):
        group = group.sort_values("Date").copy()
        vol, mdd = _future_windows(group.Close.to_numpy(float), horizon)
        group[f"future_vol_{horizon}d"] = vol
        group[f"future_mdd_{horizon}d"] = mdd
        parts.append(group[["Date", "Ticker", f"future_vol_{horizon}d", f"future_mdd_{horizon}d"]])
    return pd.concat(parts, ignore_index=True)


def attach_future_risk(
    labels: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizon: int = 120,
    risky_quantile: float = 0.30,
) -> pd.DataFrame:
    labels = labels.copy()
    outcomes = outcomes.copy()
    labels["Date"] = pd.to_datetime(labels.Date)
    outcomes["Date"] = pd.to_datetime(outcomes.Date)
    result = labels.merge(outcomes, on=["Date", "Ticker"], how="left", validate="one_to_one")
    vol, mdd = f"future_vol_{horizon}d", f"future_mdd_{horizon}d"
    result = result.dropna(subset=[vol, mdd]).copy()
    result["z_future_vol"] = result.groupby("Date")[vol].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    result["z_future_drawdown"] = result.groupby("Date")[mdd].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    result["future_risk_score"] = (result.z_future_vol - result.z_future_drawdown) / 2
    result["future_risky"] = result.groupby("Date").future_risk_score.transform(
        lambda values: values >= values.quantile(1 - risky_quantile)
    )
    result["false_stable"] = result.stable & result.future_risky
    return result


def k_sensitivity(frame: pd.DataFrame, dates: pd.DatetimeIndex, policy: StabilityPolicy) -> pd.DataFrame:
    rows = []
    source = frame.loc[frame.Date.isin(dates) & frame.eligible_signal]
    for day, group in source.groupby("Date"):
        cols = [policy.volatility_column, policy.beta_column]
        ready = group.dropna(subset=cols).copy()
        ready["abs_beta"] = ready[policy.beta_column].abs()
        x = ready[[policy.volatility_column, "abs_beta"]]
        if len(x) < 30:
            continue
        scaled = StandardScaler().fit_transform(x)
        for k in range(2, 7):
            labels = KMeans(n_clusters=k, n_init=20, random_state=policy.random_state).fit_predict(scaled)
            shares = pd.Series(labels).value_counts(normalize=True)
            rows.append(
                dict(
                    Date=day,
                    k=k,
                    silhouette=silhouette_score(scaled, labels),
                    dbi=davies_bouldin_score(scaled, labels),
                    min_cluster_share=shares.min(),
                )
            )
    return pd.DataFrame(rows)


def group_future_summary(panel: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    vol, mdd = f"future_vol_{horizon}d", f"future_mdd_{horizon}d"
    return (
        panel.groupby("stability_group")
        .agg(
            rows=("Ticker", "size"),
            dates=("Date", "nunique"),
            mean_future_vol=(vol, "mean"),
            mean_future_mdd=(mdd, "mean"),
            mdd_breach=(mdd, lambda values: values.le(threshold).mean()),
            future_risky_rate=("future_risky", "mean"),
        )
        .reset_index()
    )


def ablation_table(panel: pd.DataFrame, frame: pd.DataFrame, policy: StabilityPolicy, horizon: int) -> pd.DataFrame:
    current = frame[["Date", "Ticker", policy.volatility_column, policy.beta_column]].copy()
    base = panel.merge(current, on=["Date", "Ticker"], how="left", validate="one_to_one")
    base["abs_beta"] = base[policy.beta_column].abs()
    rows = []
    for day, group in base.groupby("Date"):
        count = int(group.stable.sum())
        flags = {
            "volatility_only": group[policy.volatility_column].rank(method="first").le(count),
            "beta_only": group.abs_beta.rank(method="first").le(count),
            "canonical_2d_kmeans": group.stable,
        }
        for name, flag in flags.items():
            chosen = group.loc[flag]
            rows.append(
                dict(
                    Date=day,
                    policy=name,
                    selected=len(chosen),
                    false_stable_rate=chosen.future_risky.mean(),
                    mean_future_mdd=chosen[f"future_mdd_{horizon}d"].mean(),
                )
            )
    return pd.DataFrame(rows)


def lookback_table(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    outcomes: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    rows = []
    for lookback in [20, 30, 60, 120, 252]:
        policy = StabilityPolicy(lookback, config.stable_k, config.random_state)
        labels = policy.label_dates(frame, dates)
        evaluated = attach_future_risk(
            labels, outcomes, config.future_risk_horizon, config.future_risky_quantile
        )
        stable = evaluated.loc[evaluated.stable]
        rows.append(
            dict(
                lookback=lookback,
                rows=len(evaluated),
                dates=evaluated.Date.nunique(),
                stable_n=len(stable),
                false_stable_rate=stable.future_risky.mean(),
                mean_future_mdd=stable[f"future_mdd_{config.future_risk_horizon}d"].mean(),
            )
        )
    return pd.DataFrame(rows)


def industry_policy(
    evaluated: pd.DataFrame,
    frame: pd.DataFrame,
    sectors: pd.DataFrame,
    policy: StabilityPolicy,
) -> pd.DataFrame:
    current = frame[["Date", "Ticker", policy.volatility_column, policy.beta_column]].copy()
    panel = evaluated.merge(current, on=["Date", "Ticker"], how="left", validate="one_to_one")
    panel = panel.merge(sectors, on="Ticker", how="left", validate="many_to_one")
    panel["Sector"] = panel.Sector.fillna("UNKNOWN")
    panel["industry_status"] = "UNKNOWN"
    for (_, sector), index in panel.groupby(["Date", "Sector"]).groups.items():
        if sector == "UNKNOWN" or len(index) < 12:
            continue
        group = panel.loc[index].copy()
        group["abs_beta"] = group[policy.beta_column].abs()
        x = group[[policy.volatility_column, "abs_beta"]]
        scaled = StandardScaler().fit_transform(x)
        model = KMeans(n_clusters=3, n_init=20, random_state=policy.random_state)
        raw = model.fit_predict(scaled)
        order = pd.DataFrame(model.cluster_centers_).mean(axis=1).sort_values().index
        mapping = {cluster: label for cluster, label in zip(order, ["STABLE", "MIDDLE", "RISKY"])}
        panel.loc[index, "industry_status"] = pd.Series(raw, index=index).map(mapping)
    panel["industry_unknown"] = panel.industry_status.eq("UNKNOWN")
    panel["balanced_verified"] = panel.stable & panel.industry_status.eq("STABLE")
    panel["balanced_eligible"] = panel.stable & (
        panel.industry_status.eq("STABLE") | panel.industry_unknown
    )
    return panel


def policy_summary(panel: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    rows = []
    for name, flag in {
        "market_stable": panel.stable,
        "balanced_verified": panel.balanced_verified,
        "balanced_eligible": panel.balanced_eligible,
    }.items():
        chosen = panel.loc[flag]
        rows.append(
            dict(
                policy=name,
                selected=len(chosen),
                dates=chosen.Date.nunique(),
                retention=len(chosen) / max(int(panel.stable.sum()), 1),
                false_stable_rate=chosen.future_risky.mean(),
                mdd_breach=chosen[f"future_mdd_{horizon}d"].le(threshold).mean(),
                mean_future_mdd=chosen[f"future_mdd_{horizon}d"].mean(),
            )
        )
    return pd.DataFrame(rows)


def block_bootstrap_policy_difference(
    panel: pd.DataFrame, n_boot: int = 2000, random_state: int = 42
) -> pd.DataFrame:
    daily = []
    for day, group in panel.groupby("Date"):
        market = group.loc[group.stable, "future_risky"].mean()
        balanced = group.loc[group.balanced_verified, "future_risky"].mean()
        if np.isfinite(market) and np.isfinite(balanced):
            daily.append((day, balanced - market))
    daily = pd.DataFrame(daily, columns=["Date", "difference"])
    if daily.empty:
        return pd.DataFrame([{"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "dates": 0}])
    rng = np.random.default_rng(random_state)
    values = daily.difference.to_numpy()
    simulated = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    return pd.DataFrame(
        [{
            "estimate": values.mean(),
            "ci_low": np.quantile(simulated, 0.025),
            "ci_high": np.quantile(simulated, 0.975),
            "dates": len(values),
        }]
    )


def _classification_metrics(y: pd.Series, pred: np.ndarray, score: np.ndarray) -> dict:
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, score) if y.nunique() > 1 else np.nan,
        "roc_auc": roc_auc_score(y, score) if y.nunique() > 1 else np.nan,
    }


def _score(model, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x)
    else:
        score = pred.astype(float)
    return pred, score


def risk_ml_tables(panel: pd.DataFrame, policy: StabilityPolicy) -> dict[str, pd.DataFrame]:
    features = [policy.volatility_column, "abs_beta"]
    data = panel.copy()
    data["abs_beta"] = data[policy.beta_column].abs()
    data = data.dropna(subset=features + ["future_risky", "future_risk_score"])
    train = data.loc[data.Date.lt("2024-01-01")]
    valid = data.loc[data.Date.between("2024-01-01", "2024-12-31")]
    test = data.loc[data.Date.between("2025-01-01", "2025-12-31")]
    if min(len(train), len(valid), len(test)) == 0:
        return {"classification": pd.DataFrame(), "regression": pd.DataFrame(), "sector": pd.DataFrame()}

    candidates = {
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, random_state=42),
    }
    validation_rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(train[features], train.future_risky.astype(int))
        pred, score = _score(model, valid[features])
        validation_rows.append({"split": "validation", "model": name, **_classification_metrics(valid.future_risky, pred, score)})
        fitted[name] = model
    validation = pd.DataFrame(validation_rows)
    guarded = validation.loc[validation.f1.ge(validation.f1.median())]
    best_name = guarded.sort_values(["recall", "pr_auc", "f1", "roc_auc"], ascending=False).iloc[0].model
    final_model = clone(candidates[best_name]).fit(
        pd.concat([train, valid])[features], pd.concat([train, valid]).future_risky.astype(int)
    )
    test_pred, test_score = _score(final_model, test[features])
    classification = pd.concat(
        [validation, pd.DataFrame([{"split": "test", "model": best_name, **_classification_metrics(test.future_risky, test_pred, test_score)}])],
        ignore_index=True,
    )

    regressors = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
    }
    reg_rows = []
    for name, model in regressors.items():
        model.fit(train[features], train.future_risk_score)
        prediction = model.predict(valid[features])
        reg_rows.append(
            dict(
                split="validation",
                model=name,
                mae=mean_absolute_error(valid.future_risk_score, prediction),
                rmse=mean_squared_error(valid.future_risk_score, prediction) ** 0.5,
                r2=r2_score(valid.future_risk_score, prediction),
            )
        )
    regression = pd.DataFrame(reg_rows)

    sector_rows = []
    train_valid = pd.concat([train, valid])
    test = test.copy()
    test["global_pred"] = test_pred
    test["global_score"] = test_score
    for sector, sector_test in test.loc[~test.Sector.eq("UNKNOWN")].groupby("Sector"):
        sector_train = train_valid.loc[train_valid.Sector.eq(sector)]
        if len(sector_train) < 50 or sector_train.future_risky.nunique() < 2:
            continue
        model = clone(candidates[best_name]).fit(sector_train[features], sector_train.future_risky.astype(int))
        sector_pred, sector_score = _score(model, sector_test[features])
        y = sector_test.future_risky.astype(int).to_numpy()
        global_pred = sector_test.global_pred.astype(int).to_numpy()
        risky = y == 1
        b = int(((global_pred[risky] == 1) & (sector_pred[risky] != 1)).sum())
        c = int(((global_pred[risky] != 1) & (sector_pred[risky] == 1)).sum())
        discordant = b + c
        sector_rows.append(
            dict(
                Sector=sector,
                actual_risky_n=int(risky.sum()),
                status="CONFIRMATORY" if risky.sum() >= 43 else "EXPLORATORY",
                global_recall=recall_score(y, global_pred, zero_division=0),
                sector_refit_recall=recall_score(y, sector_pred, zero_division=0),
                delta_recall=recall_score(y, sector_pred, zero_division=0)
                - recall_score(y, global_pred, zero_division=0),
                global_pr_auc=average_precision_score(y, sector_test.global_score),
                sector_pr_auc=average_precision_score(y, sector_score),
                discordant_pairs=discordant,
                paired_exact_p=binomtest(b, n=discordant, p=0.5).pvalue if discordant else np.nan,
            )
        )
    return {
        "classification": classification,
        "regression": regression,
        "sector": pd.DataFrame(sector_rows),
    }


def run_stability_suite(frame: pd.DataFrame, calendar: pd.DatetimeIndex, sectors: pd.DataFrame, config: ProjectConfig) -> tuple[dict[str, pd.DataFrame], StabilityPolicy]:
    policy = StabilityPolicy(config.stable_lookback, config.stable_k, config.random_state)
    dates = evaluation_dates(calendar)
    outcomes = forward_outcomes(frame, config.future_risk_horizon)
    labels = policy.label_dates(frame, dates)
    evaluated = attach_future_risk(labels, outcomes, config.future_risk_horizon, config.future_risky_quantile)
    industry = industry_policy(evaluated, frame, sectors, policy)
    handoff_columns = [
        "Date", "Ticker", "Sector", "risk_rank", "stability_group", "stable",
        "industry_status", "industry_unknown", "balanced_verified", "balanced_eligible",
        policy.volatility_column, policy.beta_column,
    ]
    tables = {
        "stability_k_sensitivity": k_sensitivity(frame, dates, policy),
        "stability_future_groups": group_future_summary(evaluated, config.future_risk_horizon, config.mdd_threshold),
        "stability_axis_ablation": ablation_table(evaluated, frame, policy, config.future_risk_horizon),
        "stability_lookback": lookback_table(frame, dates, outcomes, config),
        "stability_industry_policy": policy_summary(industry, config.future_risk_horizon, config.mdd_threshold),
        "stability_industry_bootstrap": block_bootstrap_policy_difference(industry, random_state=config.random_state),
        "stability_handoff": industry[handoff_columns].copy(),
    }
    tables.update({f"stability_risk_ml_{name}": value for name, value in risk_ml_tables(industry, policy).items()})
    return tables, policy
