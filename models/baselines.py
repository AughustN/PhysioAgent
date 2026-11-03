"""Non-LLM baselines for the T0 furosemide dose task, on cohort v2 (data/*.csv).

Arms: median, eGFR-band median (the README floor), Ridge and XGBoost. Ridge and XGBoost
run a dev-set grid search (snapped MAE), then the winner is refit on train+dev and scored
once on test. Every prediction is snapped to the dose ladder before scoring, so the
numbers are comparable with the LLM/KG arms, which answer in ladder steps.

    python -m models.baselines
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # dir containing PhysioAgent/
# The repo dir is named PhysioAgent but every module imports as physioagent.*; register
# its path under the name modules expect, so dataset.py's own imports resolve.
import types  # noqa: E402
_pkg = types.ModuleType("physioagent")
_pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("physioagent", _pkg)
from physioagent.kg.ehr.dataset import CATEGORICAL, FEATURE_COLUMNS, check_inputs  # noqa: E402

SEED = 20260928
LADDER = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0]
# Midpoints between adjacent rungs; gaps are 10 twice then 20 everywhere else.
MIDPOINTS = [15.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 180.0]
EGFR_EDGES = [0, 15, 30, 45, 1000]  # README floor bands: <=15 / <=30 / <=45 / >45


def snap(y) -> np.ndarray:
    return np.asarray(LADDER)[np.searchsorted(MIDPOINTS, y, side="left")]


def scores(y_true, y_snap, y_raw=None) -> dict:
    """MAE/MAPE on ladder-snapped preds; acc@5/@10 on RAW preds — against ladder rungs
    a miss is always a whole rung away, so rel error is 0 or >=50% and the acc buckets
    would collapse to the exact-hit rate."""
    y_true, y_snap = np.asarray(y_true), np.asarray(y_snap)
    y_raw = y_snap if y_raw is None else np.asarray(y_raw)
    return {
        "MAE": float(np.mean(np.abs(y_snap - y_true))),
        "MAPE": float(np.mean(np.abs(y_snap - y_true) / y_true)),
        "acc@5": float(np.mean(np.abs(y_raw - y_true) / y_true <= 0.05)),
        "acc@10": float(np.mean(np.abs(y_raw - y_true) / y_true <= 0.10)),
    }


def preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]),
             [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def ridge_pipeline(params: dict) -> Pipeline:
    return Pipeline([("pre", preprocess()), ("ridge", Ridge(**params))])


def xgb_pipeline(params: dict) -> Pipeline:
    return Pipeline([("pre", preprocess()),
                     ("xgb", XGBRegressor(random_state=SEED, n_jobs=-1, **params))])


def search(model_fn, grid, X_tr, y_tr, X_dev, y_dev):
    """Grid-search on dev by snapped MAE; returns (best params, best dev MAE)."""
    best, best_mae = None, float("inf")
    for params in grid:
        model = model_fn(params).fit(X_tr, y_tr)
        mae = mean_absolute_error(y_dev, snap(model.predict(X_dev)))
        print(f"    {params} -> dev MAE {mae:.2f}")
        if mae < best_mae:
            best, best_mae = params, mae
    return best, best_mae


def main() -> None:
    cohort = pd.read_csv(ROOT / "data" / "cohort.csv", dtype={"hadm_id": str})
    features = pd.read_csv(ROOT / "data" / "features.csv", dtype={"hadm_id": str})
    # dose_band is forbidden as model input but kept for the subgroup report.
    rows = cohort[["hadm_id", "dose_mg", "split", "dose_band"]].merge(
        features, on="hadm_id", validate="one_to_one")
    check_inputs(rows[FEATURE_COLUMNS].columns)
    splits = {s: rows[rows["split"] == s] for s in ("train", "dev", "test")}
    X_train = splits["train"][FEATURE_COLUMNS]
    y_train_s = splits["train"]["dose_mg"]  # Series, for band medians
    y_train = y_train_s.to_numpy()

    # eGFR-band floor: per-band median learned on train.
    band_lo = pd.cut(splits["train"]["egfr_t_dec"], EGFR_EDGES, include_lowest=True)
    band_median = y_train_s.groupby(band_lo, observed=True).median()

    def egfr_pred(split: str) -> np.ndarray:
        med = pd.cut(splits[split]["egfr_t_dec"], EGFR_EDGES, include_lowest=True)
        mapped = med.map(band_median)  # same bins -> same category labels as band_median
        return mapped.fillna(float(np.median(y_train))).to_numpy()

    y_dev, y_test = splits["dev"]["dose_mg"].to_numpy(), splits["test"]["dose_mg"].to_numpy()

    print("Ridge grid:")
    r_best, _ = search(ridge_pipeline,
                       [{"alpha": a} for a in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)],
                       X_train, y_train, splits["dev"][FEATURE_COLUMNS], y_dev)
    print("XGBoost grid:")
    x_best, _ = search(xgb_pipeline,
                       [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
                        for n in (200, 600) for d in (2, 4, 6) for lr in (0.05, 0.2)],
                       X_train, y_train, splits["dev"][FEATURE_COLUMNS], y_dev)
    print(f"best ridge {r_best} | best xgb {x_best}")

    # Refit winners on train+dev, score test once. Values are (snapped, raw) pairs;
    # median and egfr-band have no raw difference.
    tr = rows[rows["split"].isin(("train", "dev"))]
    Xtr, ytr = tr[FEATURE_COLUMNS], tr["dose_mg"].to_numpy()
    preds = {
        "median": (raw := np.full(len(y_test), float(np.median(y_train))), raw),
        "egfr-band": (snap(egfr_pred("test")), egfr_pred("test")),
        "ridge": (r := snap(rp := ridge_pipeline(r_best).fit(Xtr, ytr)
                            .predict(splits["test"][FEATURE_COLUMNS])), rp),
        "xgboost": (x := snap(xp := xgb_pipeline(x_best).fit(Xtr, ytr)
                              .predict(splits["test"][FEATURE_COLUMNS])), xp),
    }

    hdr = f"{'arm':<10}{'MAE':>8}{'MAPE':>8}{'acc@5':>8}{'acc@10':>8}{'d_med':>8}"
    print(f"\nTest, {len(y_test)} rows. MAE/MAPE on ladder-snapped preds, acc on raw. "
          f"Band medians { {str(k): round(v, 1) for k, v in band_median.items()} }")
    print(hdr)
    med = scores(y_test, preds["median"][0])
    for arm in ("median", "egfr-band", "ridge", "xgboost"):
        s = scores(y_test, *preds[arm])
        d = "" if arm == "median" else f"{s['MAE'] - med['MAE']:>+7.2f} "
        print(f"{arm:<10}{s['MAE']:>7.2f} {s['MAPE']:>7.3f} {s['acc@5']:>7.3f} "
              f"{s['acc@10']:>7.3f} {d}")

    print("\nBy dose_band (MAE on snapped preds):")
    band_test = splits["test"]["dose_band"].to_numpy()
    for b in pd.unique(band_test):
        idx = band_test == b
        line = f"  {b:<6}n={idx.sum():<4}"
        for arm in ("median", "egfr-band", "ridge", "xgboost"):
            s = scores(y_test[idx], preds[arm][0][idx])
            line += f"  {arm}:{s['MAE']:.2f}"
        print(line)


if __name__ == "__main__":
    main()
