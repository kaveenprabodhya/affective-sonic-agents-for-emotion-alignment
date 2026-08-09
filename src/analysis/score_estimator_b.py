"""H1 scoring with the held-out Estimator B, plus an integrity freeze. Read-only on stimuli.

The optimisation loop was coached by Estimator A, so A-distances only prove the optimiser
obeyed its coach. H1 requires the INDEPENDENT judge: Estimator B (different corpus, different
model family, frozen before generation). This script scores every stimulus with B, computes
the non-optimised vs optimised B-distance to each brief's target for all 48 pairs, and writes
the H1 dataset. It also hashes every file and checks for silence, freezing the stimulus set.

It never modifies the stimuli and never regenerates anything.

    python src/analysis/score_estimator_b.py
"""
import sys
import json
import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT                       # noqa: E402
from estimators.model import load as load_estimator         # noqa: E402
from estimators.data import combined_features               # noqa: E402
from features.extracts import load_audio                     # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dist(e, t):
    return float(np.hypot(e[0] - t[0], e[1] - t[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="estimator_B",
                    help="Which frozen judge to score with (default: estimator_B, "
                         "the pre-specified incumbent)")
    ap.add_argument("--suffix", default=None,
                    help="Output suffix. Defaults to '' for estimator_B so the "
                         "incumbent keeps its original filenames, otherwise the "
                         "estimator name.")
    args = ap.parse_args()
    tag = args.suffix if args.suffix is not None else (
        "" if args.estimator == "estimator_B" else f"_{args.estimator}")

    exp = load("experiment.yaml")
    sr = exp["synthesis"]["sample_rate_hz"]
    stim_dir = ROOT / "data" / "stimuli"
    manifest = json.loads((stim_dir / "manifest.json").read_text())
    briefs = {b["id"]: b for b in load("briefs.yaml")["briefs"]}
    predict_b, meta_b = load_estimator(args.estimator)
    print(f"{args.estimator}: {meta_b['corpus']}/{meta_b['model_family']}, "
          f"held_out={meta_b['held_out_from_optimisation']}")
    disc = meta_b.get("discrimination_on_study_stimuli")
    if disc:
        print(f"  selected by architecture comparison; discrimination "
              f"{disc['discrimination']:.3f}")
    print()

    integrity, rows, silent, missing = {}, [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in manifest:
            target = tuple(r["target"])
            rec = {"brief": r["brief"], "run": r["run"],
                   "quadrant": briefs[r["brief"]]["quadrant"],
                   "target_v": target[0], "target_a": target[1]}
            for cond, key in (("non_optimised", "nonopt"), ("optimised", "opt")):
                fpath = stim_dir / r[cond]["file"]
                if not fpath.exists():
                    missing.append(r[cond]["file"])
                    continue
                y, _sr = load_audio(str(fpath), sr=sr)
                peak = float(np.max(np.abs(y)))
                rms = float(np.sqrt(np.mean(y ** 2)))
                non_silent = peak > 0.01 and rms > 1e-4
                if not non_silent:
                    silent.append(r[cond]["file"])
                integrity[r[cond]["file"]] = {
                    "sha256": sha256(fpath), "peak": round(peak, 4), "rms": round(rms, 5),
                    "duration_s": round(len(y) / _sr, 2), "non_silent": non_silent}
                est_b = predict_b(combined_features(y, _sr))
                rec[f"{key}_B_v"] = round(est_b[0], 3)
                rec[f"{key}_B_a"] = round(est_b[1], 3)
                rec[f"{key}_B_dist"] = round(dist(est_b, target), 4)
                rec[f"{key}_A_dist"] = r[cond]["distance"]      # coach distance, for reference
            rec["B_reduction"] = round(rec["nonopt_B_dist"] - rec["opt_B_dist"], 4)
            rec["A_reduction"] = round(rec["nonopt_A_dist"] - rec["opt_A_dist"], 4)
            rows.append(rec)

    df = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"h1_estimator_b{tag}.csv", index=False)
    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2))

    # ---- integrity report ----
    print("=== INTEGRITY ===")
    print(f"files hashed: {len(integrity)}")
    print(f"missing files: {missing or 'none'}")
    print(f"silent files:  {silent or 'none'}")
    if missing or silent:
        print("!! Resolve integrity failures before the audience run.")

    # ---- H1 descriptive (formal paired test is Stage 6) ----
    n = len(df)
    print("\n=== H1 (Estimator B, held-out judge) - DESCRIPTIVE, all 48 pairs ===")
    print(f"mean non-optimised B-distance: {df['nonopt_B_dist'].mean():.3f}")
    print(f"mean optimised B-distance:     {df['opt_B_dist'].mean():.3f}")
    print(f"mean B-reduction (non-opt - opt): {df['B_reduction'].mean():+.3f}")
    print(f"pairs where B-distance decreased: {(df['B_reduction'] > 0).sum()}/{n}")
    print(f"pairs where B-distance increased: {(df['B_reduction'] < 0).sum()}/{n}")
    print(f"pairs unchanged (iteration-0):    {(df['A_reduction'].abs() < 1e-6).sum()}/{n}")
    # per axis, since valence is the weak axis
    dv_no = (df['nonopt_B_v'] - df['target_v']).abs().mean()
    dv_op = (df['opt_B_v'] - df['target_v']).abs().mean()
    da_no = (df['nonopt_B_a'] - df['target_a']).abs().mean()
    da_op = (df['opt_B_a'] - df['target_a']).abs().mean()
    print(f"\nper-axis mean |error|:  valence {dv_no:.3f} -> {dv_op:.3f}   "
          f"arousal {da_no:.3f} -> {da_op:.3f}")
    print(f"\n(reference) mean A-reduction (coach): {df['A_reduction'].mean():+.3f}")
    sd_v, sd_a = df[["nonopt_B_v", "opt_B_v"]].stack().std(), df[["nonopt_B_a", "opt_B_a"]].stack().std()
    rmse = meta_b["metrics_heldout_songs"]
    print(f"\njudge discrimination on these stimuli: valence SD {sd_v:.4f} "
          f"(own RMSE {rmse['valence_RMSE']}), arousal SD {sd_a:.4f} "
          f"(own RMSE {rmse['arousal_RMSE']})")
    if sd_v < 0.25 * rmse["valence_RMSE"]:
        print("  !! valence SD is under a quarter of this judge's own RMSE: it is barely")
        print("     discriminating between stimuli, so H1 cannot be fairly tested with it.")
    print(f"\nSaved: data/analysis/h1_estimator_b{tag}.csv  +  data/analysis/integrity.json")
    print("This IS the H1 dataset. Keep all 48 pairs; formal paired/mixed test is Stage 6.")


if __name__ == "__main__":
    main()