"""Architecture selection for the held-out judge.

WHY THIS EXISTS
---------------
Estimator B was originally an RBF SVR chosen to differ in model family from the
random-forest coach, not because any comparison favoured it. A diagnostic on the
frozen artefact found it produced near-constant output on the study stimuli
(valence SD 0.013 against its own held-out RMSE of 0.236), so it could not
discriminate between the artefacts it was meant to judge. This module runs the
architecture comparison that should have been run in the first place.

THE SELECTION RULE IS FIXED BEFORE ANY CANDIDATE IS FITTED
----------------------------------------------------------
Both criteria are independent of the H1 outcome. Neither looks at whether
optimised stimuli score closer to target than non-optimised ones.

  1. In-domain accuracy   - R2 on held-out songs from the estimator's own corpus.
                            Standard generalisation performance.
  2. Discrimination       - prediction SD on the study stimuli, expressed as a
                            ratio of the candidate's own RMSE. Asks whether the
                            instrument varies at all on the material it must
                            judge. A ratio near zero means a constant output.

  Gate then rank: candidates within `r2_tolerance` of the best mean in-domain R2
  pass the gate; among those, the highest discrimination wins; ties break on
  in-domain R2.

The rule lives in config/experiment.yaml under `estimator_selection`. It is
hashed into the report and the frozen metadata, so any later edit to the rule is
visible in the audit trail.

RUN
    python src/estimators/select_estimator.py --pmemo datasets/PMEmo --name estimator_B2
    python src/estimators/select_estimator.py --pmemo datasets/PMEmo --no-freeze   # report only

The incumbent is never overwritten. The winner freezes under a new name so both
can be scored and both reported.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load as load_cfg, ROOT          # noqa: E402
from estimators.data import (build_deam, build_pmemo,      # noqa: E402
                             combined_features, feature_columns)
from features.extracts import load_audio                   # noqa: E402

MODELS_DIR = ROOT / "models"
SEED = 42


# --------------------------------------------------------------------------
# candidate grid
# --------------------------------------------------------------------------
def candidates():
    """Every architecture considered, with its hyperparameter grid.

    Ridge is included deliberately: a linear model extrapolates outside its
    training manifold rather than saturating, so it is the candidate most likely
    to retain dynamic range under domain shift. Kernel and tree models are
    expected to compress. Including both is what makes the comparison informative.
    """
    out = []

    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        out.append(("ridge", {"alpha": alpha},
                    make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=SEED))))

    for C in (1.0, 10.0, 100.0):
        for gamma in ("scale", 0.01, 0.001, 0.0001):
            out.append(("svr_rbf", {"C": C, "gamma": gamma},
                        make_pipeline(StandardScaler(),
                                      MultiOutputRegressor(
                                          SVR(kernel="rbf", C=C, gamma=gamma)))))

    for C in (1.0, 10.0):
        out.append(("svr_linear", {"C": C},
                    make_pipeline(StandardScaler(),
                                  MultiOutputRegressor(SVR(kernel="linear", C=C)))))

    for depth in (None, 12, 20):
        for leaf in (1, 5):
            out.append(("rf", {"max_depth": depth, "min_samples_leaf": leaf},
                        RandomForestRegressor(n_estimators=400, max_depth=depth,
                                              min_samples_leaf=leaf,
                                              random_state=SEED, n_jobs=-1)))

    for n_est in (200, 400):
        for lr in (0.05, 0.1):
            out.append(("gbr", {"n_estimators": n_est, "learning_rate": lr},
                        MultiOutputRegressor(
                            GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                                      max_depth=3, random_state=SEED))))

    for hidden in ((128, 64), (64,)):
        for alpha in (1e-4, 1e-2):
            out.append(("mlp", {"hidden_layer_sizes": hidden, "alpha": alpha},
                        make_pipeline(StandardScaler(),
                                      MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                                                   max_iter=1000, early_stopping=True,
                                                   random_state=SEED))))
    return out


# --------------------------------------------------------------------------
# splits and metrics
# --------------------------------------------------------------------------
def three_way_split(df, feat_cols):
    """Split by song, never by window, so no song appears in two partitions.

    train fits the candidate, validation ranks hyperparameters within a family,
    test provides the reported in-domain metric. Keeping tuning off the test
    partition is what stops the reported R2 being optimistic.
    """
    X = df[feat_cols].to_numpy()
    y = df[["valence", "arousal"]].to_numpy()
    g = df["song_id"].to_numpy()

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    rest_i, test_i = next(outer.split(X, y, g))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_rel, val_rel = next(inner.split(X[rest_i], y[rest_i], g[rest_i]))
    return X, y, rest_i[tr_rel], rest_i[val_rel], test_i


def score(model, X, y, idx):
    p = model.predict(X[idx])
    return {
        "valence_R2": float(r2_score(y[idx, 0], p[:, 0])),
        "valence_RMSE": float(mean_squared_error(y[idx, 0], p[:, 0]) ** 0.5),
        "arousal_R2": float(r2_score(y[idx, 1], p[:, 1])),
        "arousal_RMSE": float(mean_squared_error(y[idx, 1], p[:, 1]) ** 0.5),
        "n_windows": int(len(idx)),
    }


# --------------------------------------------------------------------------
# study stimuli
# --------------------------------------------------------------------------
def stimulus_features(feat_cols, sr):
    """Extract features for every stimulus the judge will have to score."""
    stim_dir = ROOT / "data" / "stimuli"
    manifest_path = stim_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest at {manifest_path}. Generation must run before selection, "
                 "because discrimination is measured on the study stimuli.")

    files = []
    for r in json.loads(manifest_path.read_text()):
        files += [r["non_optimised"]["file"], r["optimised"]["file"]]
    files = sorted(set(files))

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in files:
            p = stim_dir / fn
            if not p.exists():
                continue
            y, _sr = load_audio(str(p), sr=sr)
            rows.append(combined_features(y, _sr))
    if not rows:
        sys.exit("No stimulus audio found. Run generation first.")
    df = pd.DataFrame(rows)
    return df[feat_cols].to_numpy(), len(rows)


def domain_shift(X_train, X_stim):
    """How far outside the training distribution the stimuli sit.

    Reported as a diagnostic, never selected on. Large values explain why kernel
    and tree models compress: an RBF kernel evaluates to ~0 against every support
    vector once the input is far enough away, leaving only the intercept.
    """
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd == 0] = 1.0
    z = (X_stim - mu) / sd
    return {
        "mean_abs_z": float(np.abs(z).mean()),
        "median_abs_z": float(np.median(np.abs(z))),
        "max_abs_z": float(np.abs(z).max()),
        "frac_features_beyond_3sd": float((np.abs(z) > 3).mean()),
        "mean_euclidean_z_distance": float(np.linalg.norm(z, axis=1).mean()),
    }


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def rule_hash(rule):
    return hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()[:12]


def apply_rule(results, rule):
    """Gate on in-domain R2, then rank on discrimination. Never touches H1."""
    tol = rule["r2_tolerance"]
    best_r2 = max(r["mean_R2"] for r in results)
    gated = [r for r in results if r["mean_R2"] >= best_r2 - tol]

    floor = rule.get("min_discrimination")
    passing = [r for r in gated if floor is None or r["discrimination"] >= floor]
    pool = passing or gated          # if nothing clears the floor, say so but still pick

    pool.sort(key=lambda r: (-r["discrimination"], -r["mean_R2"]))
    return pool[0], gated, bool(passing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmemo", help="Path to PMEmo root (Estimator B corpus)")
    ap.add_argument("--deam", help="Path to DEAM root (only if --corpus DEAM)")
    ap.add_argument("--corpus", default="PMEmo", choices=["PMEmo", "DEAM"])
    ap.add_argument("--name", default="estimator_B2", help="Name to freeze the winner under")
    ap.add_argument("--role", default="held_out_H1_judge")
    ap.add_argument("--songs", type=int, default=None, help="Cap songs (quick run)")
    ap.add_argument("--no-freeze", action="store_true", help="Report only, freeze nothing")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.name in ("estimator_A", "estimator_B"):
        sys.exit("Refusing to overwrite a frozen estimator. Choose a new --name.")

    exp = load_cfg("experiment.yaml")
    rule = exp.get("estimator_selection")
    if not rule:
        sys.exit("No `estimator_selection` block in config/experiment.yaml. The selection "
                 "rule must be recorded before candidates are fitted.")
    sr = exp["synthesis"]["sample_rate_hz"]
    rhash = rule_hash(rule)

    print(f"Selection rule [{rhash}]: gate within {rule['r2_tolerance']} R2 of best, "
          f"then rank by discrimination"
          + (f" (floor {rule['min_discrimination']})" if rule.get("min_discrimination") else ""))
    print("Criteria are independent of H1. The H1 comparison is not computed here.\n")

    if args.corpus == "PMEmo":
        if not args.pmemo:
            sys.exit("--pmemo required")
        df = build_pmemo(args.pmemo, n_songs=args.songs, use_cache=not args.no_cache)
    else:
        if not args.deam:
            sys.exit("--deam required")
        df = build_deam(args.deam, n_songs=args.songs, use_cache=not args.no_cache)

    feat_cols = feature_columns(df)
    X, y, tr, val, te = three_way_split(df, feat_cols)
    print(f"{args.corpus}: {len(df)} windows, {df['song_id'].nunique()} songs "
          f"-> train {len(tr)} / val {len(val)} / test {len(te)} (split by song)\n")

    X_stim, n_stim = stimulus_features(feat_cols, sr)
    shift = domain_shift(X[tr], X_stim)
    print(f"Study stimuli: {n_stim} files")
    print(f"Domain shift vs training distribution: mean |z| = {shift['mean_abs_z']:.2f}, "
          f"{shift['frac_features_beyond_3sd']*100:.1f}% of features beyond 3 SD\n")

    # ---- fit every candidate, pick the best hyperparameters per family on val ----
    print(f"Fitting {len(candidates())} candidate configurations...")
    per_config = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (family, params, model) in enumerate(candidates(), 1):
            try:
                model.fit(X[tr], y[tr])
            except Exception as e:                      # a config that cannot fit is not a candidate
                print(f"  [{i:2d}/{len(candidates())}] {family} {params} FAILED: {e}")
                continue
            v = score(model, X, y, val)
            per_config.append({"family": family, "params": params, "model": model,
                               "val_mean_R2": (v["valence_R2"] + v["arousal_R2"]) / 2})
            print(f"  [{i:2d}/{len(candidates())}] {family:11s} "
                  f"val mean R2 = {per_config[-1]['val_mean_R2']:+.3f}  {params}")

    if not per_config:
        sys.exit("No candidate fitted successfully.")

    best_per_family = {}
    for c in per_config:
        cur = best_per_family.get(c["family"])
        if cur is None or c["val_mean_R2"] > cur["val_mean_R2"]:
            best_per_family[c["family"]] = c

    # ---- score family winners on the untouched test split + on the stimuli ----
    print("\nEvaluating family winners on the held-out test split and the study stimuli...")
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for family, c in sorted(best_per_family.items()):
            m = c["model"]
            t = score(m, X, y, te)
            p_stim = m.predict(X_stim)
            sd_v, sd_a = float(p_stim[:, 0].std()), float(p_stim[:, 1].std())
            ratio_v = sd_v / t["valence_RMSE"] if t["valence_RMSE"] else 0.0
            ratio_a = sd_a / t["arousal_RMSE"] if t["arousal_RMSE"] else 0.0
            results.append({
                "family": family, "params": c["params"],
                "valence_R2": t["valence_R2"], "valence_RMSE": t["valence_RMSE"],
                "arousal_R2": t["arousal_R2"], "arousal_RMSE": t["arousal_RMSE"],
                "mean_R2": (t["valence_R2"] + t["arousal_R2"]) / 2,
                "stim_sd_valence": sd_v, "stim_sd_arousal": sd_a,
                "stim_range_valence": float(np.ptp(p_stim[:, 0])),
                "stim_range_arousal": float(np.ptp(p_stim[:, 1])),
                "range_ratio_valence": ratio_v, "range_ratio_arousal": ratio_a,
                "discrimination": min(ratio_v, ratio_a),
                "_model": m,
            })

    winner, gated, cleared_floor = apply_rule(results, rule)

    # ---- report ----
    out_dir = MODELS_DIR / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in results])
    tab["gated_in"] = tab["family"].isin([g["family"] for g in gated])
    tab["selected"] = tab["family"] == winner["family"]
    tab = tab.sort_values("discrimination", ascending=False)
    tab.to_csv(out_dir / "candidate_comparison.csv", index=False)

    lines = [
        "ESTIMATOR ARCHITECTURE SELECTION",
        "=" * 72,
        f"corpus:        {args.corpus}",
        f"selection rule: [{rhash}] gate within {rule['r2_tolerance']} mean R2 of best, "
        f"then maximise discrimination",
        f"criteria:      in-domain R2 (held-out songs) and discrimination "
        f"(stimulus prediction SD / own RMSE)",
        "H1 was not consulted at any point in this procedure.",
        "",
        "DOMAIN SHIFT (diagnostic, not a selection criterion)",
        "-" * 72,
    ]
    for k, v in shift.items():
        lines.append(f"  {k:32s} {v:.3f}")
    lines += [
        "",
        "CANDIDATES (best hyperparameters per family, scored on the untouched test split)",
        "-" * 72,
        f"{'family':12s} {'val R2':>7s} {'aro R2':>7s} {'mean R2':>8s} "
        f"{'sd_v':>7s} {'sd_a':>7s} {'discrim':>8s}  gate  sel",
    ]
    for r in tab.to_dict("records"):
        lines.append(f"{r['family']:12s} {r['valence_R2']:+7.3f} {r['arousal_R2']:+7.3f} "
                     f"{r['mean_R2']:+8.3f} {r['stim_sd_valence']:7.4f} "
                     f"{r['stim_sd_arousal']:7.4f} {r['discrimination']:8.3f}  "
                     f"{'yes' if r['gated_in'] else ' no':>4s}  "
                     f"{'<<<' if r['selected'] else '   '}")
    lines += [
        "",
        "DISCRIMINATION is the prediction SD on the 96 study stimuli divided by the",
        "candidate's own held-out RMSE. A value near zero means the estimator returns",
        "almost the same coordinate whatever it is given, and cannot judge H1 at all.",
        "",
        f"SELECTED: {winner['family']}  {winner['params']}",
        f"  in-domain   valence R2 {winner['valence_R2']:+.3f}  "
        f"arousal R2 {winner['arousal_R2']:+.3f}",
        f"  on stimuli  valence SD {winner['stim_sd_valence']:.4f}  "
        f"arousal SD {winner['stim_sd_arousal']:.4f}",
        f"  discrimination {winner['discrimination']:.3f}",
    ]
    if not cleared_floor and rule.get("min_discrimination"):
        lines += [
            "",
            f"WARNING: no candidate reached the discrimination floor of "
            f"{rule['min_discrimination']}.",
            "The winner is the best available, but every architecture compressed on this",
            "stimulus set. That is evidence the limitation is the domain gap rather than",
            "the choice of model family, and H1 should be reported as untestable with any",
            "of these judges rather than as rejected.",
        ]
    report = "\n".join(lines)
    (out_dir / "selection_report.txt").write_text(report + "\n")
    print("\n" + report)

    # ---- freeze ----
    if args.no_freeze:
        print(f"\n--no-freeze: nothing written to models/. Report in {out_dir}/")
        return

    final = None
    for family, params, model in candidates():
        if family == winner["family"] and params == winner["params"]:
            final = model
            break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)                      # refit on the full corpus; metrics already honest

    joblib.dump({"model": final, "feature_names": feat_cols},
                MODELS_DIR / f"{args.name}.joblib")

    meta = {
        "name": args.name,
        "corpus": args.corpus,
        "model_family": winner["family"],
        "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in winner["params"].items()},
        "role": args.role,
        "held_out_from_optimisation": True,
        "selected_by": "architecture comparison, criteria independent of H1",
        "selection_rule": rule,
        "selection_rule_hash": rhash,
        "n_candidates_fitted": len(per_config),
        "families_compared": sorted(best_per_family),
        "feature_set": "combined_A_plus_B",
        "n_features": len(feat_cols),
        "output_scale": [-1, 1],
        "n_windows": int(len(df)), "n_songs": int(df["song_id"].nunique()),
        "metrics_heldout_songs": {
            "valence_R2": round(winner["valence_R2"], 3),
            "valence_RMSE": round(winner["valence_RMSE"], 3),
            "arousal_R2": round(winner["arousal_R2"], 3),
            "arousal_RMSE": round(winner["arousal_RMSE"], 3),
        },
        "discrimination_on_study_stimuli": {
            "stim_sd_valence": round(winner["stim_sd_valence"], 4),
            "stim_sd_arousal": round(winner["stim_sd_arousal"], 4),
            "range_ratio_valence": round(winner["range_ratio_valence"], 3),
            "range_ratio_arousal": round(winner["range_ratio_arousal"], 3),
            "discrimination": round(winner["discrimination"], 3),
        },
        "domain_shift_vs_training": {k: round(v, 3) for k, v in shift.items()},
        "sklearn_version": sklearn.__version__,
        "frozen_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
    }
    (MODELS_DIR / f"{args.name}.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nFroze models/{args.name}.joblib  (+ .meta.json)")
    print(f"Comparison table: models/selection/candidate_comparison.csv")
    print(f"\nThe incumbent estimator_B is untouched. Score both and report both:")
    print(f"  python src/analysis/score_estimator_b.py --estimator estimator_B")
    print(f"  python src/analysis/score_estimator_b.py --estimator {args.name}")


if __name__ == "__main__":
    main()