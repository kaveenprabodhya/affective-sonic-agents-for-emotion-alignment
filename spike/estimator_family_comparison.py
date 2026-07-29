"""
Stage 0 spike -- feature-family comparison
==========================================
The transfer test showed length is not the constraint, but valence was weak. That
estimator used family A (spectral/timbral) only, which lacks the tonal features
(mode, chroma, tonnetz) that carry musical valence. This asks whether valence lifts
when family B (tonal/rhythmic/dynamic) is added.

For each short window length it extracts BOTH families once, then trains three
estimators -- A only, B only, A+B combined -- on the SAME song-level split, and
reports valence and arousal accuracy. If valence rises materially with tonal
features, the A/B feature design should be revised (relax strict disjointness, since
dual-corpus training already carries the independence guarantee). If it barely moves,
valence is a documented ceiling.

Run:
    python spike/estimator_family_comparison.py --deam datasets/DEAM --songs 200
"""
from __future__ import annotations
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
import estimator_transfer_test as tt          # reuse validated DEAM loading/windowing
from features.extracts import (load_audio, extract_estimator_a,   # noqa: E402
                              extract_estimator_b)

LENGTHS = [2.0, 3.0, 5.0]      # short-clip focus
WINDOWS_PER_SONG = 6
TEST_FRACTION = 0.25
SEED = 42


def build_dataset_both(deam_root: str, n_songs: int) -> pd.DataFrame:
    v_df, v_times = tt._load_dynamic(tt._dynamic_paths(deam_root)[0])
    a_df, a_times = tt._load_dynamic(tt._dynamic_paths(deam_root)[1])
    song_ids = [s for s in v_df.index if s in a_df.index][:n_songs]
    rows = []
    for n, sid in enumerate(song_ids, 1):
        apath = tt._audio_path(deam_root, sid)
        if apath is None:
            continue
        try:
            y, sr = load_audio(apath)
        except Exception as e:
            print(f"  skip {sid}: {e}")
            continue
        dur = len(y) / sr
        v_row = v_df.loc[sid].to_numpy(dtype=float)
        a_row = a_df.loc[sid].to_numpy(dtype=float)
        annot_end = min(dur, float(v_times.max()) + 0.5)
        for length in LENGTHS:
            for t0 in tt._even_starts(tt.ANNOT_START_S, annot_end, length, WINDOWS_PER_SONG):
                seg = y[int(t0 * sr): int((t0 + length) * sr)]
                if seg.size < int(0.5 * sr):
                    continue
                val = tt._label(v_row, v_times, t0, length)
                aro = tt._label(a_row, a_times, t0, length)
                if np.isnan(val) or np.isnan(aro):
                    continue
                row = {"song_id": sid, "length": length, "valence": val, "arousal": aro}
                row.update({f"a__{k}": v for k, v in extract_estimator_a(seg, sr).items()})
                row.update({f"b__{k}": v for k, v in extract_estimator_b(seg, sr).items()})
                rows.append(row)
        if n % 25 == 0:
            print(f"  processed {n}/{len(song_ids)} songs, {len(rows)} windows so far")
    return pd.DataFrame(rows)


def evaluate_families(df: pd.DataFrame) -> pd.DataFrame:
    a_cols = [c for c in df.columns if c.startswith("a__")]
    b_cols = [c for c in df.columns if c.startswith("b__")]
    sets = {
        "A (spectral/timbral)": a_cols,
        "B (tonal/rhythmic/dyn)": b_cols,
        "A+B (combined)": a_cols + b_cols,
    }
    out = []
    for length in sorted(df["length"].unique()):
        sub = df[df["length"] == length]
        if sub["song_id"].nunique() < 4 or len(sub) < 20:
            continue
        groups = sub["song_id"].to_numpy()
        y = sub[["valence", "arousal"]].to_numpy()
        # one split, reused across all three feature sets
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                        random_state=SEED).split(np.zeros(len(sub)), y, groups))
        for name, cols in sets.items():
            X = sub[cols].to_numpy()
            model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            out.append({
                "length_s": length,
                "feature_set": name,
                "n_windows": len(sub),
                "val_R2": round(r2_score(y[te, 0], pred[:, 0]), 3),
                "val_RMSE": round(mean_squared_error(y[te, 0], pred[:, 0]) ** 0.5, 3),
                "aro_R2": round(r2_score(y[te, 1], pred[:, 1]), 3),
                "aro_RMSE": round(mean_squared_error(y[te, 1], pred[:, 1]) ** 0.5, 3),
            })
    return pd.DataFrame(out)


def plot(results: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths = sorted(results["length_s"].unique())
    sets = list(results["feature_set"].unique())
    shades = {sets[0]: "0.2", sets[1]: "0.5", sets[2]: "0.8"}
    hatches = {sets[0]: "", sets[1]: "//", sets[2]: ".."}
    x = np.arange(len(lengths))
    w = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in [(ax1, "val_R2", "Valence R$^2$"),
                              (ax2, "aro_R2", "Arousal R$^2$")]:
        for j, s in enumerate(sets):
            vals = [results[(results.length_s == L) & (results.feature_set == s)][metric].iloc[0]
                    for L in lengths]
            ax.bar(x + (j - 1) * w, vals, w, label=s, color=shades[s],
                   edgecolor="black", hatch=hatches[s])
        ax.set_xticks(x); ax.set_xticklabels([f"{int(L)}s" for L in lengths])
        ax.set_xlabel("Window length"); ax.set_ylabel("R$^2$ (held-out songs)")
        ax.set_title(title)
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"\nSaved figure: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deam", required=True)
    ap.add_argument("--songs", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Building dataset ({args.songs} songs, both feature families)...")
    print("Note: family B uses CQT-based features, so this is slower than the transfer test.")
    df = build_dataset_both(args.deam, args.songs)
    if df.empty:
        print("No windows built -- check the DEAM path."); return
    print(f"\nTotal windows: {len(df)} across {df['song_id'].nunique()} songs")
    results = evaluate_families(df)
    print("\n=== FEATURE-FAMILY COMPARISON (short clips) ===")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.out, "family_comparison.csv"), index=False)
    plot(results, os.path.join(args.out, "family_comparison.png"))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()