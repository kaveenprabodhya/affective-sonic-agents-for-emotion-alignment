"""Build windowed (features, valence, arousal) datasets for the estimators.

Both estimators use the COMBINED A+B feature set; they differ only in corpus
(DEAM vs PMEmo) and model family, which is what secures their independence.

Labels sit on a common [-1, 1] scale: DEAM annotations are native; PMEmo labels
are linearly rescaled (scale auto-detected and reported). Extracted features are
cached to disk so retraining does not re-extract.
"""
from __future__ import annotations
import os
import glob
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.extracts import load_audio, extract_estimator_a, extract_estimator_b  # noqa: E402

WINDOW_S = 3.0            # matches the sonic-logo length
WINDOWS_PER_SONG = 6
DEAM_ANNOT_START_S = 15.0  # DEAM continuous annotations begin at 15 s
CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "cache"


def combined_features(y, sr) -> dict:
    fa = extract_estimator_a(y, sr)
    fb = extract_estimator_b(y, sr)
    return {**{f"a__{k}": v for k, v in fa.items()},
            **{f"b__{k}": v for k, v in fb.items()}}


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith(("a__", "b__"))]


def _even_starts(t0, t1, length, k):
    last = t1 - length
    if last < t0:
        return []
    n = int((last - t0) // length) + 1
    if n <= k:
        return [t0 + i * length for i in range(n)]
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return [t0 + i * length for i in idx]


def _cache_path(tag: str, window_s: float, n_songs) -> Path:
    key = hashlib.md5(f"{tag}|{window_s}|{n_songs}".encode()).hexdigest()[:10]
    return CACHE_DIR / f"{tag}_{window_s}s_{key}.pkl"


# --------------------------- DEAM (dynamic labels, native [-1,1]) ---------------------------

def _deam_paths(root):
    base = os.path.join(root, "DEAM_Annotations", "annotations",
                        "annotations averaged per song", "dynamic (per second annotations)")
    return os.path.join(base, "valence.csv"), os.path.join(base, "arousal.csv")


def _load_deam_dynamic(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("song_id")
    cols = [c for c in df.columns if c.startswith("sample_")]
    times = np.array([int(c.replace("sample_", "").replace("ms", "")) / 1000.0 for c in cols])
    order = np.argsort(times)
    return df[[cols[i] for i in order]], times[order]


def _label_window(vals, times, t0, length):
    m = (times >= t0) & (times < t0 + length)
    seg = vals[m]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if seg.size else np.nan


def build_deam(root, n_songs=None, window_s=WINDOW_S, use_cache=True) -> pd.DataFrame:
    cache = _cache_path("deam", window_s, n_songs)
    if use_cache and cache.exists():
        print(f"  [cache] {cache.name}")
        return pd.read_pickle(cache)

    v_df, v_t = _load_deam_dynamic(_deam_paths(root)[0])
    a_df, a_t = _load_deam_dynamic(_deam_paths(root)[1])
    ids = [s for s in v_df.index if s in a_df.index]
    if n_songs:
        ids = ids[:n_songs]

    rows = []
    for n, sid in enumerate(ids, 1):
        ap = os.path.join(root, "DEAM_audio", "MEMD_audio", f"{sid}.mp3")
        if not os.path.exists(ap):
            continue
        try:
            y, sr = load_audio(ap)
        except Exception as e:
            print(f"  skip {sid}: {e}")
            continue
        dur = len(y) / sr
        v_row, a_row = v_df.loc[sid].to_numpy(float), a_df.loc[sid].to_numpy(float)
        end = min(dur, float(v_t.max()) + 0.5)
        for t0 in _even_starts(DEAM_ANNOT_START_S, end, window_s, WINDOWS_PER_SONG):
            seg = y[int(t0 * sr):int((t0 + window_s) * sr)]
            if seg.size < int(0.5 * sr):
                continue
            val = _label_window(v_row, v_t, t0, window_s)
            aro = _label_window(a_row, a_t, t0, window_s)
            if np.isnan(val) or np.isnan(aro):
                continue
            rows.append({"song_id": sid, "valence": val, "arousal": aro,
                         **combined_features(seg, sr)})
        if n % 50 == 0:
            print(f"  DEAM {n}/{len(ids)} songs, {len(rows)} windows")

    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    print(f"  DEAM done: {len(df)} windows, {df['song_id'].nunique()} songs. Cached {cache.name}")
    return df


# --------------------------- PMEmo (dynamic 0.5s labels, scale auto-detected) ---------------------------

def _find_one(root, *names):
    for n in names:
        hit = glob.glob(os.path.join(root, "**", n), recursive=True)
        if hit:
            return hit[0]
    return None


def _load_pmemo_dynamic_wide(path):
    """Wide dynamic-label CSV: musicId + N comma-separated values, one per 0.5s frame.
    (The header names only one value column, but each data row is genuinely wide, and
    rows vary in length because clips differ in duration.)
    Returns {musicId: np.array(values)}; per-song frame times are derived on use."""
    per = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            mid = int(float(parts[0]))
            per[mid] = np.array([float(x) for x in parts[1:] if x != ""], dtype=float)
    return per


def _rescale_to_unit(x: np.ndarray):
    """Detect the source scale and report the mapping onto [-1, 1].
    Order matters: test [0,1] and [1,9] BEFORE the already-[-1,1] case, because
    [0,1] values also fall numerically inside [-1,1] and must still be remapped."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    if -0.05 <= lo and hi <= 1.05:
        src = (0.0, 1.0)
    elif 0.95 <= lo and hi <= 9.05:
        src = (1.0, 9.0)
    elif -1.05 <= lo and hi <= 1.05:
        src = (-1.0, 1.0)
    else:
        src = (float(lo), float(hi))
    return None, src


def _pmemo_audio(root, mid):
    for m in {mid, int(mid) if str(mid).replace('.', '', 1).isdigit() else mid}:
        for pat in (f"Chorus/{m}.mp3", f"**/{m}.mp3", f"**/{m}.wav"):
            hit = glob.glob(os.path.join(root, pat), recursive=True)
            if hit:
                return hit[0]
    return None


def build_pmemo(root, n_songs=None, window_s=WINDOW_S, use_cache=True) -> pd.DataFrame:
    cache = _cache_path("pmemo", window_s, n_songs)
    if use_cache and cache.exists():
        print(f"  [cache] {cache.name}")
        return pd.read_pickle(cache)

    v_path = _find_one(root, "V_dynamic_mean.csv")
    a_path = _find_one(root, "A_dynamic_mean.csv")
    if not (v_path and a_path):
        raise FileNotFoundError("PMEmo V_dynamic_mean.csv / A_dynamic_mean.csv not found.")
    print(f"  PMEmo dynamic labels: {os.path.basename(v_path)}, {os.path.basename(a_path)}")
    v_per = _load_pmemo_dynamic_wide(v_path)
    a_per = _load_pmemo_dynamic_wide(a_path)

    allv = np.concatenate([x[~np.isnan(x)] for x in v_per.values() if x.size])
    alla = np.concatenate([x[~np.isnan(x)] for x in a_per.values() if x.size])
    _, v_src = _rescale_to_unit(allv)
    _, a_src = _rescale_to_unit(alla)
    print(f"  PMEmo valence scale {v_src} -> [-1,1]; arousal scale {a_src} -> [-1,1]")

    def to_unit(x, src):
        return float(np.clip(2 * (x - src[0]) / (src[1] - src[0]) - 1, -1, 1))

    ids = [m for m in v_per if m in a_per]
    if n_songs:
        ids = ids[:n_songs]

    rows = []
    for n, mid in enumerate(ids, 1):
        ap = _pmemo_audio(root, mid)
        if ap is None:
            continue
        try:
            y, sr = load_audio(ap)
        except Exception as e:
            print(f"  skip {mid}: {e}")
            continue
        dur = len(y) / sr
        v_row, a_row = v_per[mid], a_per[mid]
        v_t = np.arange(len(v_row)) * 0.5            # per-song time axis
        a_t = np.arange(len(a_row)) * 0.5
        end = min(dur, float(v_t.max()) + 0.5) if v_t.size else dur
        for t0 in _even_starts(0.0, end, window_s, WINDOWS_PER_SONG):
            seg = y[int(t0 * sr):int((t0 + window_s) * sr)]
            if seg.size < int(0.5 * sr):
                continue
            val = _label_window(v_row, v_t, t0, window_s)
            aro = _label_window(a_row, a_t, t0, window_s)
            if np.isnan(val) or np.isnan(aro):
                continue
            rows.append({"song_id": mid, "valence": to_unit(val, v_src),
                         "arousal": to_unit(aro, a_src), **combined_features(seg, sr)})
        if n % 50 == 0:
            print(f"  PMEmo {n}/{len(ids)} songs, {len(rows)} windows")

    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    print(f"  PMEmo done: {len(df)} windows, {df['song_id'].nunique()} songs. Cached {cache.name}")
    return df