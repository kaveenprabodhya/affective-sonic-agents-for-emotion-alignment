"""
Stage 0 feasibility spike
=========================
Question: does valence/arousal (VA) estimation transfer from long excerpts to the
short (2-5 s) clips our sonic logos will be?

Method: using DEAM's averaged per-second dynamic annotations, cut each excerpt into
windows of several lengths, label each window with the mean VA over that span, extract
the Estimator-A (spectral/timbral) feature family, and train a provisional regressor.
Accuracy is reported per window length. A SONG-LEVEL train/test split is used so that
windows from the same song never appear in both train and test (which would inflate
the scores). If accuracy at 2-5 s collapses relative to the long baseline, that is the
no-go signal for feature-mediated VA estimation on short stimuli.

This is a provisional single estimator for feasibility only -- not the frozen
Estimators A/B, which are built and frozen later (Stage 3).

Run:
    python spike/estimator_transfer_test.py --deam /path/to/DEAM --songs 200
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.extracts import load_audio, extract_estimator_a, SR  # noqa: E402

WINDOW_LENGTHS = [2.0, 3.0, 5.0, 10.0, 30.0]   # seconds; 30 s ~= long baseline
WINDOWS_PER_SONG = 6                            # cap per length, evenly spaced
ANNOT_START_S = 15.0                            # DEAM annotations begin at 15 s
RAMP_DISCARD_S = 0.0                            # extra lead-in to drop (0 = none extra)
TEST_FRACTION = 0.25
SEED = 42


# ---------- DEAM annotation loading ------------------------------------------

def _dynamic_paths(deam_root: str) -> tuple[str, str]:
    base = os.path.join(deam_root, "DEAM_Annotations", "annotations",
                        "annotations averaged per song",
                        "dynamic (per second annotations)")
    return os.path.join(base, "valence.csv"), os.path.join(base, "arousal.csv")


def _load_dynamic(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a wide dynamic-annotation CSV. Returns (df indexed by song_id,
    array of sample times in seconds)."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("song_id")
    sample_cols = [c for c in df.columns if c.startswith("sample_")]
    times = np.array([int(c.replace("sample_", "").replace("ms", "")) / 1000.0
                      for c in sample_cols])
    order = np.argsort(times)
    return df[[sample_cols[i] for i in order]], times[order]


def _audio_path(deam_root: str, song_id: int) -> str | None:
    p = os.path.join(deam_root, "DEAM_audio", "MEMD_audio", f"{song_id}.mp3")
    return p if os.path.exists(p) else None


# ---------- windowing + labelling --------------------------------------------

def _even_starts(t0: float, t1: float, length: float, k: int) -> list[float]:
    """Up to k non-overlapping window starts evenly spread across [t0, t1-length]."""
    last = t1 - length
    if last < t0:
        return []
    n_max = int((last - t0) // length) + 1
    if n_max <= k:
        return [t0 + i * length for i in range(n_max)]
    idx = np.linspace(0, n_max - 1, k).round().astype(int)
    return [t0 + i * length for i in np.unique(idx)]


def _label(vals: np.ndarray, times: np.ndarray, t0: float, length: float):
    m = (times >= t0) & (times < t0 + length)
    if not m.any():
        return np.nan
    seg = vals[m]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if seg.size else np.nan


def build_dataset(deam_root: str, n_songs: int):
    v_df, v_times = _load_dynamic(_dynamic_paths(deam_root)[0])
    a_df, a_times = _load_dynamic(_dynamic_paths(deam_root)[1])
    song_ids = [s for s in v_df.index if s in a_df.index][:n_songs]

    rows = []
    v_all, a_all = [], []
    for n, sid in enumerate(song_ids, 1):
        apath = _audio_path(deam_root, sid)
        if apath is None:
            continue
        try:
            y, sr = load_audio(apath)
        except Exception as e:
            print(f"  skip {sid}: load error ({e})")
            continue
        dur = len(y) / sr
        v_row = v_df.loc[sid].to_numpy(dtype=float)
        a_row = a_df.loc[sid].to_numpy(dtype=float)
        v_all.append(v_row[~np.isnan(v_row)])
        a_all.append(a_row[~np.isnan(a_row)])
        annot_end = min(dur, float(v_times.max()) + 0.5)
        start = ANNOT_START_S + RAMP_DISCARD_S
        for length in WINDOW_LENGTHS:
            for t0 in _even_starts(start, annot_end, length, WINDOWS_PER_SONG):
                seg = y[int(t0 * sr): int((t0 + length) * sr)]
                if seg.size < int(0.5 * sr):
                    continue
                val = _label(v_row, v_times, t0, length)
                aro = _label(a_row, a_times, t0, length)
                if np.isnan(val) or np.isnan(aro):
                    continue
                feats = extract_estimator_a(seg, sr)
                rows.append({"song_id": sid, "length": length,
                             "valence": val, "arousal": aro, **feats})
        if n % 25 == 0:
            print(f"  processed {n}/{len(song_ids)} songs, {len(rows)} windows so far")

    scale = (np.concatenate(v_all) if v_all else np.array([]),
             np.concatenate(a_all) if a_all else np.array([]))
    return pd.DataFrame(rows), scale


# ---------- per-length training + evaluation ---------------------------------

def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in df.columns
                 if c not in ("song_id", "length", "valence", "arousal")]
    out = []
    for length in sorted(df["length"].unique()):
        sub = df[df["length"] == length]
        n_songs = sub["song_id"].nunique()
        if n_songs < 4 or len(sub) < 20:
            out.append({"length_s": length, "n_windows": len(sub),
                        "n_songs": n_songs, "note": "too few to model"})
            continue
        X = sub[feat_cols].to_numpy()
        y = sub[["valence", "arousal"]].to_numpy()
        groups = sub["song_id"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
        model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        out.append({
            "length_s": length,
            "n_windows": len(sub),
            "n_songs": n_songs,
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
    r = results.dropna(subset=["val_R2"]) if "val_R2" in results else results
    if r.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.plot(r["length_s"], r["val_R2"], marker="o", color="black", label="valence")
    ax1.plot(r["length_s"], r["aro_R2"], marker="s", linestyle="--",
             color="0.45", label="arousal")
    ax1.set_xlabel("Window length (s)"); ax1.set_ylabel("R$^2$ (held-out songs)")
    ax1.set_title("Accuracy vs clip length"); ax1.legend(frameon=False)
    ax2.plot(r["length_s"], r["val_RMSE"], marker="o", color="black", label="valence")
    ax2.plot(r["length_s"], r["aro_RMSE"], marker="s", linestyle="--",
             color="0.45", label="arousal")
    ax2.set_xlabel("Window length (s)"); ax2.set_ylabel("RMSE (annotation units)")
    ax2.set_title("Error vs clip length"); ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"\nSaved figure: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deam", required=True, help="Path to the DEAM root folder")
    ap.add_argument("--songs", type=int, default=200, help="Number of songs to use")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Building dataset from {args.songs} songs...")
    df, (v_scale, a_scale) = build_dataset(args.deam, args.songs)
    if df.empty:
        print("No windows built -- check the DEAM path and structure."); return
    print(f"\nTotal windows: {len(df)} across {df['song_id'].nunique()} songs")
    if v_scale.size:
        print(f"Valence annotation range: [{v_scale.min():.2f}, {v_scale.max():.2f}] "
              f"mean {v_scale.mean():.2f}")
        print(f"Arousal annotation range: [{a_scale.min():.2f}, {a_scale.max():.2f}] "
              f"mean {a_scale.mean():.2f}")

    results = evaluate(df)
    print("\n=== TRANSFER TEST: accuracy by window length ===")
    print(results.to_string(index=False))
    df.to_csv(os.path.join(args.out, "transfer_windows.csv"), index=False)
    results.to_csv(os.path.join(args.out, "transfer_results.csv"), index=False)
    plot(results, os.path.join(args.out, "estimator_transfer.png"))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()