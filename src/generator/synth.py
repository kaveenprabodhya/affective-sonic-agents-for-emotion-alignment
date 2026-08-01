"""Parametric sonic-logo synthesis: acoustic parameters -> MIDI motif -> WAV.

Deterministic and interpretable: the LLM proposes a small set of named acoustic
parameters, this module renders them, so a parameter change maps to an auditable
change in the sound (the reason for parametric over neural generation).
"""
from __future__ import annotations
import os
import tempfile
import subprocess
import numpy as np
import pretty_midi

# Curated General MIDI instruments (name -> GM program) spanning timbres.
INSTRUMENTS = {
    "acoustic grand piano": 0, "celesta": 8, "music box": 10, "vibraphone": 11,
    "marimba": 12, "tubular bells": 14, "church organ": 19, "nylon guitar": 24,
    "orchestral harp": 46, "string ensemble": 48, "choir": 52, "trumpet": 56,
    "french horn": 60, "flute": 73, "ocarina": 79, "new age pad": 88, "warm pad": 89,
}

SCHEMA = {
    "tempo_bpm": ("int", 40, 200),
    "mode": ("enum", ["major", "minor"]),
    "pitch_center_midi": ("int", 48, 84),
    "pitch_range": ("int", 2, 24),
    "contour": ("enum", ["rising", "falling", "arch", "flat", "wave"]),
    "notes_per_beat": ("enum", [1, 2, 4]),
    "dynamics": ("enum", ["soft", "moderate", "loud"]),
    "articulation": ("enum", ["legato", "staccato"]),
    "instrument": ("enum", list(INSTRUMENTS)),
}
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
_VELOCITY = {"soft": 50, "moderate": 80, "loud": 110}


def schema_text() -> str:
    """Human-readable schema block to embed in the generator prompt."""
    lines = []
    for k, spec in SCHEMA.items():
        if spec[0] == "int":
            lines.append(f'  "{k}": integer {spec[1]}-{spec[2]}')
        else:
            vals = ", ".join(str(v) for v in spec[1])
            lines.append(f'  "{k}": one of [{vals}]')
    return "{\n" + ",\n".join(lines) + "\n}"


def validate_params(p: dict):
    """Return (clean_params, None) or (None, reason). Numerics clamped; enums must match."""
    if not isinstance(p, dict):
        return None, "not a JSON object"
    out = {}
    for k, spec in SCHEMA.items():
        if k not in p:
            return None, f"missing {k}"
        v = p[k]
        if spec[0] == "int":
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                return None, f"{k} not a number"
            out[k] = int(np.clip(v, spec[1], spec[2]))
        else:
            allowed = spec[1]
            if isinstance(v, str):
                v = v.strip().lower()
            if v not in allowed:
                return None, f"{k}={p[k]!r} not one of {allowed}"
            out[k] = v
    return out, None


def _contour_indices(shape, pool_size, n_notes):
    n, m = max(1, n_notes), pool_size
    if m <= 1:
        return [0] * n
    if shape == "rising":
        idx = np.linspace(0, m - 1, n)
    elif shape == "falling":
        idx = np.linspace(m - 1, 0, n)
    elif shape == "arch":
        half = np.linspace(0, m - 1, (n + 1) // 2)
        idx = np.concatenate([half, half[::-1][1:]])[:n]
    elif shape == "wave":
        idx = (m - 1) / 2 * (1 + np.sin(np.linspace(0, 2 * np.pi, n)))
    else:  # flat
        idx = np.full(n, m // 2)
    return [int(round(i)) for i in idx]


def params_to_midi(p: dict, duration: float = 3.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(p["tempo_bpm"]))
    inst = pretty_midi.Instrument(program=INSTRUMENTS[p["instrument"]])
    scale = MAJOR if p["mode"] == "major" else MINOR
    center, rng = p["pitch_center_midi"], p["pitch_range"]
    lo, hi = center - rng // 2, center + rng // 2
    pool = sorted(n for n in range(lo, hi + 1) if (n - center) % 12 in scale) or [center]

    spacing = (60.0 / p["tempo_bpm"]) / p["notes_per_beat"]
    n_notes = max(1, int(duration / spacing))
    dur_ratio = 0.9 if p["articulation"] == "legato" else 0.4
    vel = _VELOCITY[p["dynamics"]]

    t = 0.0
    for idx in _contour_indices(p["contour"], len(pool), n_notes):
        if t >= duration:
            break
        pitch = pool[idx % len(pool)]
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=int(pitch), start=t,
            end=min(duration, t + spacing * dur_ratio)))
        t += spacing
    pm.instruments.append(inst)
    return pm


def render(p: dict, soundfont: str, out_wav: str, duration: float = 3.0, sr: int = 22050):
    """Render parameters to a WAV via FluidSynth, trimmed to exactly `duration`.

    The raw FluidSynth output carries a reverb/release tail past the notes, so it is
    cut to `duration` with a short fade-out, then peak-normalised to a level set by
    the `dynamics` parameter (so loudness reflects the parameter, not the patch).
    """
    import soundfile as sf
    pm = params_to_midi(p, duration)
    fd_m, mid = tempfile.mkstemp(suffix=".mid")
    os.close(fd_m)
    fd_w, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd_w)
    try:
        pm.write(mid)
        cmd = ["fluidsynth", "-ni", "-g", "1.0", "-r", str(sr), "-T", "wav",
               "-F", tmp_wav, soundfont, mid]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp_wav):
            raise RuntimeError(f"fluidsynth failed: {r.stderr.strip()[:300]}")
        y, _ = sf.read(tmp_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = int(duration * sr)
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        fade = min(int(0.06 * sr), len(y))
        y[-fade:] *= np.linspace(1.0, 0.0, fade)
        target = {"soft": 0.4, "moderate": 0.7, "loud": 0.95}[p["dynamics"]]
        peak = float(np.max(np.abs(y))) or 1.0
        y = (y * (target / peak)).astype(np.float32)
        sf.write(out_wav, y, sr)
    finally:
        for f in (mid, tmp_wav):
            if os.path.exists(f):
                os.unlink(f)
    return out_wav