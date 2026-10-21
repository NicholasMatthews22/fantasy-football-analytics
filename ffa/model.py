"""Projection model: gradient-boosted trees for the projection, plus a calibrated
floor (15th percentile outcome) and ceiling (85th percentile outcome).

The floor/ceiling come from how actual scores spread around the model's projections
on a season it didn't train on, measured separately for each position."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import CURRENT_SEASON, METRICS_PATH, MODEL_DIR, POSITIONS
from .features import FEATURES

QUANTILES = {"floor": 0.15, "ceiling": 0.85}
RELEVANT = 6.0   # evaluate on fantasy-relevant players (recent form of 6+ PPR points)


def _model():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.04,
        max_leaf_nodes=24, min_samples_leaf=60, l2_regularization=1.0,
        categorical_features=[FEATURES.index("pos_code")], random_state=7)


def _range_ratios(df):
    """Hold out the latest season, then measure actual/projected at the 15th and 85th percentile."""
    last = df["season"].max()
    early, late = df[df["season"] < last], df[df["season"] == last]
    m = _model().fit(early[FEATURES], early["ppr"])
    late = late.assign(p=m.predict(late[FEATURES]))
    late = late[late["p"] >= RELEVANT]
    ratios = {}
    for pos in POSITIONS:
        r = (late.loc[late["position"] == pos, "ppr"] / late.loc[late["position"] == pos, "p"])
        ratios[pos] = (float(r.quantile(QUANTILES["floor"])), float(r.quantile(QUANTILES["ceiling"])))
    return ratios


def fit(df):
    return {"proj": _model().fit(df[FEATURES], df["ppr"]), "ratios": _range_ratios(df)}


def predict(models, df):
    out = pd.DataFrame(index=df.index)
    out["proj"] = np.clip(models["proj"].predict(df[FEATURES]), 0, None)
    lo = df["position"].map({k: v[0] for k, v in models["ratios"].items()})
    hi = df["position"].map({k: v[1] for k, v in models["ratios"].items()})
    out["floor"] = out["proj"] * lo
    out["ceiling"] = out["proj"] * hi
    return out


def _score(test, preds):
    """Mean absolute error of the model vs. a simple baseline (recent average)."""
    t = test.assign(**preds)
    t = t[t["ewm_ppr"] >= RELEVANT]
    rows = []
    for pos in ["ALL"] + POSITIONS:
        s = t if pos == "ALL" else t[t["position"] == pos]
        rows.append({
            "position": pos, "games": int(len(s)),
            "model_mae": round(float((s["proj"] - s["ppr"]).abs().mean()), 2),
            "baseline_mae": round(float((s["ewm_ppr"] - s["ppr"]).abs().mean()), 2),
            "range_hit_rate": round(float(((s["ppr"] >= s["floor"]) & (s["ppr"] <= s["ceiling"])).mean()), 3),
            "rank_corr": round(float(s[["proj", "ppr"]].corr("spearman").iloc[0, 1]), 3),
        })
    return rows


def backtest(df):
    """Two honest tests: train on earlier seasons, predict a season the model never saw.
    Also returns last season's held-out predictions, for the head-to-head test against Sleeper."""
    results, holdout = {}, None
    tests = {f"{CURRENT_SEASON - 1} season": CURRENT_SEASON - 1,
             f"{CURRENT_SEASON} season so far": CURRENT_SEASON}
    for label, season in tests.items():
        train, test = df[df["season"] < season], df[df["season"] == season]
        if len(test) == 0:
            continue
        preds = predict(fit(train), test)
        results[label] = _score(test, preds)
        if season == CURRENT_SEASON - 1:
            holdout = test[["season", "week", "player_id", "position", "ewm_ppr", "ppr"]].join(preds[["proj"]])
    return results, holdout


def train_and_save(df):
    metrics, holdout = backtest(df)
    models = fit(df)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_DIR / "models.joblib")
    imp = feature_importance(models["proj"], df)
    METRICS_PATH.write_text(json.dumps({"backtest": metrics, "importance": imp,
                                        "train_rows": int(len(df))}, indent=2))
    return models, metrics, holdout


def feature_importance(model, df, n=4000):
    """Permutation importance on a sample: how much worse the model gets if a feature is scrambled."""
    from sklearn.inspection import permutation_importance
    s = df.sample(min(n, len(df)), random_state=1)
    r = permutation_importance(model, s[FEATURES], s["ppr"], n_repeats=3, random_state=1,
                               scoring="neg_mean_absolute_error")
    imp = sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(v), 3)} for f, v in imp]


def load_models():
    return joblib.load(MODEL_DIR / "models.joblib")


def walk_forward(df, season, weeks):
    """Re-create the projections we *would* have made before each week, using only earlier data.

    Used to backfill the accuracy tracker for weeks before live tracking began.
    """
    out = []
    for w in weeks:
        before = (df["season"] < season) | ((df["season"] == season) & (df["week"] < w))
        target = df[(df["season"] == season) & (df["week"] == w)]
        if len(target) == 0:
            continue
        preds = predict(fit(df[before]), target)
        out.append(target[["player_id", "name", "position", "team", "season", "week", "ewm_ppr"]].join(preds))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
