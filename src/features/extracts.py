"""
Acoustic feature extraction for the sonic-logo emotional-alignment pipeline.

The module provides three feature views:

  1. extract_estimator_a()    -- spectral and timbral features, including MFCCs
                                 and spectral-shape statistics.

  2. extract_estimator_b()    -- rhythmic, tonal and dynamic features, including
                                 tempo, chroma, tonnetz and energy statistics.

     These functions define two feature families for extraction and organisation.
     Downstream training combines both families and supplies the same combined
     feature vector to Estimators A and B. Estimator independence therefore
     comes from their different training datasets and model families, not from
     feature-level separation.

  3. extract_audience_block() -- a smaller, interpretable and emotionally neutral
                                 descriptor set supplied to the OCEAN audience
                                 agents as a key:value block. It contains no target
                                 coordinates, estimator outputs or evaluative labels.

All feature views use the same load_audio() function. Audio is loaded in mono at
a fixed sample rate for reproducibility.
"""
from __future__ import annotations
import numpy as np
import librosa

SR = 22050          # fixed sample rate (Hz)
N_FFT = 2048
HOP = 512
FMIN_PITCH = librosa.note_to_hz("C2")   # ~65 Hz
FMAX_PITCH = librosa.note_to_hz("C7")   # ~2093 Hz

# Krumhansl-Kessler key profiles (major, minor); used to estimate mode.
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono waveform at the fixed sample rate."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


# ----- shared low-level computations -----------------------------------------

def _tempo(y, sr):
    return float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])


def _onset_rate(y, sr):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    dur = len(y) / sr
    return float(len(onsets) / dur) if dur > 0 else 0.0


def _rms_db(y):
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    return 20.0 * np.log10(rms + 1e-8)


def _pitch_track(y, sr):
    """Return (mean_midi, slope_midi_per_s) over voiced frames, or (nan, nan)."""
    f0, _, _ = librosa.pyin(y, fmin=FMIN_PITCH, fmax=FMAX_PITCH, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    m = ~np.isnan(f0)
    if m.sum() < 2:
        return float("nan"), float("nan")
    midi = librosa.hz_to_midi(f0[m])
    slope = float(np.polyfit(times[m], midi, 1)[0])
    return float(np.mean(midi)), slope


def _mode_and_key(y, sr):
    """Estimate major/minor mode by Krumhansl-Kessler profile correlation.
    Returns (mode, tonic_pitch_class)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best_r, best_mode, best_pc = -np.inf, "major", 0
    for shift in range(12):
        c = np.roll(chroma, -shift)
        rmaj = np.corrcoef(c, _KK_MAJOR)[0, 1]
        rmin = np.corrcoef(c, _KK_MINOR)[0, 1]
        if rmaj > best_r:
            best_r, best_mode, best_pc = rmaj, "major", shift
        if rmin > best_r:
            best_r, best_mode, best_pc = rmin, "minor", shift
    return best_mode, best_pc


# ----- 1. audience numeric block ---------------------------------------------

def extract_audience_block(y: np.ndarray, sr: int = SR) -> dict:
    """Small, interpretable, emotionally-neutral descriptor set for the agents."""
    dur = len(y) / sr
    mode, _ = _mode_and_key(y, sr)
    mean_midi, slope = _pitch_track(y, sr)
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    db = _rms_db(y)
    return {
        "duration_s": round(dur, 2),
        "tempo_bpm": round(_tempo(y, sr), 1),
        "mode": mode,
        "mean_pitch_midi": None if np.isnan(mean_midi) else round(mean_midi, 1),
        "pitch_slope_midi_per_s": None if np.isnan(slope) else round(slope, 3),
        "spectral_centroid_hz": round(centroid, 1),
        "rms_energy": round(rms, 4),
        "onset_rate_per_s": round(_onset_rate(y, sr), 2),
        "dynamic_range_db": round(float(np.percentile(db, 95) - np.percentile(db, 5)), 1),
    }


def format_audience_block(block: dict) -> str:
    """Render the audience block as a neutral key:value text block for prompts."""
    labels = {
        "duration_s": "Duration (s)",
        "tempo_bpm": "Tempo (BPM)",
        "mode": "Mode",
        "mean_pitch_midi": "Mean pitch (MIDI)",
        "pitch_slope_midi_per_s": "Pitch slope (MIDI/s)",
        "spectral_centroid_hz": "Spectral centroid (Hz)",
        "rms_energy": "RMS energy",
        "onset_rate_per_s": "Onset rate (per s)",
        "dynamic_range_db": "Dynamic range (dB)",
    }
    return "\n".join(f"{labels[k]}: {block[k]}" for k in labels)


# ----- 2. spectral / timbral feature family ---------------------------------

def extract_estimator_a(y: np.ndarray, sr: int = SR) -> dict:
    """Extract the spectral and timbral feature family used in the combined estimator input."""
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i+1}_std"] = float(mfcc[i].std())
    for name, fn, needs_sr in [
        ("centroid", librosa.feature.spectral_centroid, True),
        ("bandwidth", librosa.feature.spectral_bandwidth, True),
        ("rolloff", librosa.feature.spectral_rolloff, True),
        ("flatness", librosa.feature.spectral_flatness, False),
    ]:
        v = fn(y=y, sr=sr) if needs_sr else fn(y=y)
        feats[f"{name}_mean"] = float(v.mean())
        feats[f"{name}_std"] = float(v.std())
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f"contrast{i+1}_mean"] = float(contrast[i].mean())
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())
    return feats


# ----- 3. rhythmic / tonal / dynamic feature family -------------------------

def extract_estimator_b(y: np.ndarray, sr: int = SR) -> dict:
    """Extract the rhythmic, tonal and dynamic feature family used in the combined estimator input."""
    feats = {"tempo_bpm": _tempo(y, sr), "onset_rate": _onset_rate(y, sr)}
    pulse = librosa.beat.plp(y=y, sr=sr)          # predominant local pulse
    feats["pulse_mean"] = float(pulse.mean())
    feats["pulse_std"] = float(pulse.std())
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = float(chroma[i].mean())
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(tonnetz.shape[0]):
        feats[f"tonnetz{i+1}_mean"] = float(tonnetz[i].mean())
    mode, tonic = _mode_and_key(y, sr)
    feats["mode_major"] = 1.0 if mode == "major" else 0.0
    feats["tonic_pc"] = float(tonic)
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    db = _rms_db(y)
    feats["dynamic_range_db"] = float(np.percentile(db, 95) - np.percentile(db, 5))
    return feats


if __name__ == "__main__":
    import sys
    y, sr = load_audio(sys.argv[1])
    print("AUDIENCE BLOCK\n" + format_audience_block(extract_audience_block(y, sr)))
    print(f"\nEstimator A family: {len(extract_estimator_a(y, sr))} features")
    print(f"Estimator B family: {len(extract_estimator_b(y, sr))} features")