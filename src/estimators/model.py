"""Train, evaluate, freeze and load a valence-arousal estimator.

Model families (kept different across A and B so they cannot share model-level bias):
  'rf'  -> RandomForest (bagged trees)          -- Estimator A
  'svr' -> RBF SVR in a scaling pipeline        -- Estimator B
  'mlp' -> small MLP (alternative for B)

Freezing writes the fitted pipeline plus a metadata sidecar (corpus, features,
family, scale, metrics, versions, date) so the estimator is a fixed, auditable
artefact. B additionally records that it is held out from the optimisation loop.
"""
from __future__ import annotations
import json
import sklearn
import datetime as _dt
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SEED = 42


def make_model(family: str):
    if family == "rf":
        return RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
    if family == "svr":
        return make_pipeline(StandardScaler(),
                             MultiOutputRegressor(SVR(kernel="rbf", C=10.0, gamma="scale")))
    if family == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800,
                                          random_state=SEED))
    raise ValueError(f"unknown model family: {family}")


def _split(df, feat_cols, test_fraction=0.2):
    X = df[feat_cols].to_numpy()
    y = df[["valence", "arousal"]].to_numpy()
    groups = df["song_id"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                    random_state=SEED).split(X, y, groups))
    return X, y, tr, te


def evaluate(model, X, y, te):
    pred = model.predict(X[te])
    return {
        "valence_R2": round(r2_score(y[te, 0], pred[:, 0]), 3),
        "valence_RMSE": round(mean_squared_error(y[te, 0], pred[:, 0]) ** 0.5, 3),
        "arousal_R2": round(r2_score(y[te, 1], pred[:, 1]), 3),
        "arousal_RMSE": round(mean_squared_error(y[te, 1], pred[:, 1]) ** 0.5, 3),
        "n_test_windows": int(len(te)),
    }


def train_and_freeze(df, feat_cols, name, corpus, family, role, held_out=False):
    """Fit on a train split, report held-out metrics, refit on all data, and freeze."""
    X, y, tr, te = _split(df, feat_cols)
    model = make_model(family)
    model.fit(X[tr], y[tr])
    metrics = evaluate(model, X, y, te)

    # refit on the full corpus for the frozen artefact (metrics already estimated honestly)
    final = make_model(family)
    final.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": final, "feature_names": feat_cols}, model_path)

    meta = {
        "name": name, "corpus": corpus, "model_family": family, "role": role,
        "held_out_from_optimisation": held_out,
        "feature_set": "combined_A_plus_B", "n_features": len(feat_cols),
        "output_scale": [-1, 1],
        "n_windows": int(len(df)), "n_songs": int(df["song_id"].nunique()),
        "metrics_heldout_songs": metrics,
        "sklearn_version": sklearn.__version__,
        "frozen_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
    }
    (MODELS_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  froze {model_path.name}  |  "
          f"valence R2={metrics['valence_R2']} arousal R2={metrics['arousal_R2']}")
    return meta


def load(name: str):
    """Load a frozen estimator. Returns (predict_fn, meta)."""
    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
    meta = json.loads((MODELS_DIR / f"{name}.meta.json").read_text())
    model, feat_names = bundle["model"], bundle["feature_names"]

    def predict(feature_dict: dict):
        """feature_dict must be a combined_features(...) dict. Returns (valence, arousal) in [-1,1]."""
        x = np.array([[feature_dict[f] for f in feat_names]])
        v, a = model.predict(x)[0]
        return float(np.clip(v, -1, 1)), float(np.clip(a, -1, 1))

    return predict, meta