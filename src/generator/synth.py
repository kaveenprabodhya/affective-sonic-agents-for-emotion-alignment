"""Parametric sonic-logo synthesis: acoustic parameters -> MIDI motif -> WAV.

Deterministic and interpretable: the LLM proposes a small set of named acoustic
parameters, this module renders them, so a parameter change maps to an auditable
change in the sound (the reason for parametric over neural generation).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pretty_midi


# Curated General MIDI instruments (name -> GM program) spanning timbres.
INSTRUMENTS = {
    "acoustic grand piano": 0,
    "celesta": 8,
    "music box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "tubular bells": 14,
    "church organ": 19,
    "nylon guitar": 24,
    "orchestral harp": 46,
    "string ensemble": 48,
    "choir": 52,
    "trumpet": 56,
    "french horn": 60,
    "flute": 73,
    "ocarina": 79,
    "new age pad": 88,
    "warm pad": 89,
}


MAJOR = [
    0,
    2,
    4,
    5,
    7,
    9,
    11,
]

MINOR = [
    0,
    2,
    3,
    5,
    7,
    8,
    10,
]


_VELOCITY = {
    "soft": 50,
    "moderate": 80,
    "loud": 110,
}


# ---------------------------------------------------------------------------
# Rhythm
# ---------------------------------------------------------------------------
#
# Inter-onset multipliers have mean ~= 1.0, so notes_per_beat still controls
# overall density while rhythm_pattern controls temporal phrasing.
#
# Accent values provide small expressive changes around the dynamics baseline.
#
RHYTHM_PATTERNS = {
    "even": {
        "ioi": [1.0],
        "accent": [1.0],
    },

    "short_short_long": {
        "ioi": [
            0.6,
            0.6,
            1.8,
        ],
        "accent": [
            1.00,
            0.90,
            1.15,
        ],
    },

    "long_short_short": {
        "ioi": [
            1.8,
            0.6,
            0.6,
        ],
        "accent": [
            1.15,
            0.90,
            1.00,
        ],
    },

    "syncopated": {
        "ioi": [
            0.75,
            1.25,
            0.75,
            1.25,
        ],
        "accent": [
            1.00,
            1.12,
            0.90,
            1.08,
        ],
    },

    "pickup_resolution": {
        "ioi": [
            0.5,
            0.5,
            1.0,
            2.0,
        ],
        "accent": [
            0.85,
            0.95,
            1.05,
            1.18,
        ],
    },

    "cadential": {
        "ioi": [
            0.7,
            0.7,
            0.7,
            1.9,
        ],
        "accent": [
            1.00,
            0.90,
            0.95,
            1.18,
        ],
    },
}


# ---------------------------------------------------------------------------
# Melodic motif
# ---------------------------------------------------------------------------
#
# contour controls the broad trajectory of the phrase:
# rising / falling / arch / flat / wave.
#
# motif_pattern controls local melodic movement around that trajectory.
#
# Values are offsets in SCALE DEGREES, not semitones. Keeping them centred
# approximately around zero avoids motif_pattern simply shifting the complete
# melody upward or downward.
#
MOTIF_PATTERNS = {
    # Small neighbouring-scale movement.
    "stepwise": [
        0,
        1,
        0,
        -1,
    ],

    # Leap away from the contour anchor, then return.
    "skip_return": [
        0,
        2,
        0,
        -2,
    ],

    # Alternating third-like movement.
    "thirds": [
        -1,
        1,
        -1,
        1,
    ],

    # Strong leap away from and back to an anchor.
    "anchor_leap": [
        0,
        3,
        0,
        -3,
    ],

    # Alternating directional motion.
    "zigzag": [
        0,
        2,
        -1,
        1,
        -2,
        1,
    ],

    # Chord-like expansion around the contour anchor.
    "arpeggio": [
        -2,
        0,
        2,
        0,
    ],
}


# ---------------------------------------------------------------------------
# Generator schema
# ---------------------------------------------------------------------------

SCHEMA = {
    "tempo_bpm": (
        "int",
        40,
        200,
    ),

    "mode": (
        "enum",
        [
            "major",
            "minor",
        ],
    ),

    "pitch_center_midi": (
        "int",
        48,
        84,
    ),

    "pitch_range": (
        "int",
        2,
        24,
    ),

    "contour": (
        "enum",
        [
            "rising",
            "falling",
            "arch",
            "flat",
            "wave",
        ],
    ),

    "motif_pattern": (
        "enum",
        list(
            MOTIF_PATTERNS
        ),
    ),

    "notes_per_beat": (
        "enum",
        [
            1,
            2,
            4,
        ],
    ),

    "rhythm_pattern": (
        "enum",
        list(
            RHYTHM_PATTERNS
        ),
    ),

    "dynamics": (
        "enum",
        [
            "soft",
            "moderate",
            "loud",
        ],
    ),

    "articulation": (
        "enum",
        [
            "legato",
            "staccato",
        ],
    ),

    "instrument": (
        "enum",
        list(
            INSTRUMENTS
        ),
    ),
}


def schema_text() -> str:
    """Human-readable schema block to embed in the generator prompt."""

    lines = []

    for k, spec in SCHEMA.items():

        if spec[0] == "int":

            lines.append(
                f'  "{k}": integer '
                f'{spec[1]}-{spec[2]}'
            )

        else:

            vals = ", ".join(
                str(v)
                for v in spec[1]
            )

            lines.append(
                f'  "{k}": one of '
                f'[{vals}]'
            )

    return (
        "{\n"
        + ",\n".join(lines)
        + "\n}"
    )


def validate_params(p: dict):
    """Return (clean_params, None) or (None, reason).

    Numeric parameters are clamped to their legal bounds.
    Enum parameters must exactly match an allowed value.
    """

    if not isinstance(
        p,
        dict,
    ):
        return (
            None,
            "not a JSON object",
        )

    out = {}

    for k, spec in SCHEMA.items():

        if k not in p:
            return (
                None,
                f"missing {k}",
            )

        v = p[k]

        if spec[0] == "int":

            try:
                v = int(
                    round(
                        float(v)
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                return (
                    None,
                    f"{k} not a number",
                )

            out[k] = int(
                np.clip(
                    v,
                    spec[1],
                    spec[2],
                )
            )

        else:

            allowed = spec[1]

            if isinstance(
                v,
                str,
            ):
                v = (
                    v
                    .strip()
                    .lower()
                )

            if v not in allowed:

                return (
                    None,
                    (
                        f"{k}={p[k]!r} "
                        f"not one of {allowed}"
                    ),
                )

            out[k] = v

    return (
        out,
        None,
    )


# ---------------------------------------------------------------------------
# Melody construction
# ---------------------------------------------------------------------------

def _contour_indices(
    shape,
    pool_size,
    n_notes,
):
    """Return exactly n_notes broad-contour pitch-pool indices."""

    n = max(
        1,
        int(n_notes),
    )

    m = max(
        1,
        int(pool_size),
    )

    if m <= 1:
        return [0] * n

    if n == 1:
        return [
            m // 2
        ]

    x = np.linspace(
        0.0,
        1.0,
        n,
    )

    if shape == "rising":

        pos = x

    elif shape == "falling":

        pos = (
            1.0
            - x
        )

    elif shape == "arch":

        # Low -> high -> low.
        pos = (
            1.0
            - np.abs(
                2.0 * x
                - 1.0
            )
        )

    elif shape == "wave":

        # Low -> high -> low
        # over one complete cycle.
        pos = (
            0.5
            * (
                1.0
                + np.sin(
                    2.0
                    * np.pi
                    * x
                    - np.pi / 2.0
                )
            )
        )

    else:  # flat

        pos = np.full(
            n,
            0.5,
        )

    idx = np.rint(
        pos
        * (m - 1)
    ).astype(int)

    return idx.tolist()


def _motif_indices(
    contour_indices,
    motif_pattern,
    pool_size,
):
    """Apply local melodic motion around a broad contour.

    contour_indices determines the overall trajectory of the phrase.

    motif_pattern adds local movement in scale degrees around that
    trajectory. Results remain inside the available scale pool.
    """

    m = max(
        1,
        int(pool_size),
    )

    if m <= 1:

        return [
            0
            for _ in contour_indices
        ]

    offsets = MOTIF_PATTERNS[
        motif_pattern
    ]

    out = []

    for i, base_idx in enumerate(
        contour_indices
    ):

        offset = offsets[
            i % len(offsets)
        ]

        idx = int(
            np.clip(
                int(base_idx)
                + int(offset),
                0,
                m - 1,
            )
        )

        out.append(
            idx
        )

    return out


# ---------------------------------------------------------------------------
# MIDI generation
# ---------------------------------------------------------------------------

def params_to_midi(
    p: dict,
    duration: float = 3.0,
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI(
        initial_tempo=float(
            p["tempo_bpm"]
        )
    )

    inst = pretty_midi.Instrument(
        program=INSTRUMENTS[
            p["instrument"]
        ]
    )

    scale = (
        MAJOR
        if p["mode"] == "major"
        else MINOR
    )

    center = p[
        "pitch_center_midi"
    ]

    rng = p[
        "pitch_range"
    ]

    lo = (
        center
        - rng // 2
    )

    hi = (
        center
        + rng // 2
    )

    pool = sorted(
        n
        for n in range(
            lo,
            hi + 1,
        )
        if (
            n - center
        ) % 12 in scale
    ) or [
        center
    ]


    # -----------------------------------------------------------------------
    # Overall note density
    # -----------------------------------------------------------------------

    base_spacing = (
        60.0
        / float(
            p["tempo_bpm"]
        )
        / float(
            p["notes_per_beat"]
        )
    )


    # -----------------------------------------------------------------------
    # Rhythm
    # -----------------------------------------------------------------------

    rhythm = RHYTHM_PATTERNS[
        p["rhythm_pattern"]
    ]

    ioi_pattern = rhythm[
        "ioi"
    ]

    accent_pattern = rhythm[
        "accent"
    ]


    # Construct variable onset positions first.
    onsets = []
    iois = []

    t = 0.0
    i = 0

    while (
        t < duration - 0.05
        and len(onsets) < 96
    ):

        multiplier = ioi_pattern[
            i
            % len(
                ioi_pattern
            )
        ]

        ioi = max(
            0.04,
            base_spacing
            * float(multiplier),
        )

        onsets.append(
            t
        )

        iois.append(
            ioi
        )

        t += ioi
        i += 1


    # -----------------------------------------------------------------------
    # Broad contour
    # -----------------------------------------------------------------------

    broad_contour = _contour_indices(
        p["contour"],
        len(pool),
        len(onsets),
    )


    # -----------------------------------------------------------------------
    # Local melodic motif
    # -----------------------------------------------------------------------
    #
    # p.get() is intentional:
    # old diagnostic parameter dictionaries created before motif_pattern was
    # added can still be rendered using the stepwise fallback.
    #
    melody = _motif_indices(
        broad_contour,
        p.get(
            "motif_pattern",
            "stepwise",
        ),
        len(pool),
    )


    # -----------------------------------------------------------------------
    # Articulation
    # -----------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # articulation controls note gate length.
    # rhythm_pattern controls time until the NEXT onset.
    #
    # Previously note duration was proportional directly to the entire IOI.
    # Therefore a "long" rhythmic interval became a long sustained note and
    # the intended rest was almost inaudible.
    #
    # The duration is now capped against the base spacing, allowing a long
    # rhythm interval to produce an audible gap.
    #
    gate_ratio = (
        0.85
        if p["articulation"]
        == "legato"
        else 0.40
    )


    base_velocity = _VELOCITY[
        p["dynamics"]
    ]


    # -----------------------------------------------------------------------
    # Notes
    # -----------------------------------------------------------------------

    for i, (
        start,
        ioi,
        idx,
    ) in enumerate(
        zip(
            onsets,
            iois,
            melody,
        )
    ):

        remaining = (
            duration
            - start
        )

        if remaining < 0.04:
            continue


        pitch = pool[
            idx
            % len(pool)
        ]


        accent = accent_pattern[
            i
            % len(
                accent_pattern
            )
        ]


        velocity = int(
            np.clip(
                round(
                    base_velocity
                    * accent
                ),
                1,
                127,
            )
        )


        # A long IOI now creates a real rhythmic gap instead of a nearly
        # continuous sustained note.
        note_duration = max(
            0.04,
            min(
                ioi * 0.90,
                base_spacing
                * gate_ratio,
            ),
        )


        end = min(
            duration,
            start
            + note_duration,
        )


        inst.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=int(pitch),
                start=float(start),
                end=float(end),
            )
        )


    pm.instruments.append(
        inst
    )

    return pm


# ---------------------------------------------------------------------------
# Reachable-region / estimator-selection grid
# ---------------------------------------------------------------------------

def grid_params(
    n=300,
    seed=0,
):
    """A deterministic spread across the parameter space.

    Lives here rather than in probe_reachable so it can be imported without
    pulling in the estimator stack.

    Both the reachable-region probe and the coach-selection probe therefore
    sample from the same synthesis definition.
    """

    import itertools


    tempos = [
        50,
        90,
        130,
        170,
        200,
    ]


    modes = [
        "major",
        "minor",
    ]


    centers = [
        50,
        62,
        74,
        84,
    ]


    contours = [
        "rising",
        "falling",
        "arch",
        "flat",
        "wave",
    ]


    motifs = list(
        MOTIF_PATTERNS
    )


    npbs = [
        1,
        2,
        4,
    ]


    rhythms = list(
        RHYTHM_PATTERNS
    )


    dyns = [
        "soft",
        "moderate",
        "loud",
    ]


    arts = [
        "legato",
        "staccato",
    ]


    insts = [
        "warm pad",
        "string ensemble",
        "nylon guitar",
        "vibraphone",
        "flute",
        "trumpet",
        "music box",
        "church organ",
    ]


    full = list(
        itertools.product(
            tempos,
            modes,
            centers,
            contours,
            motifs,
            npbs,
            rhythms,
            dyns,
            arts,
            insts,
        )
    )


    rng = np.random.default_rng(
        seed
    )


    idx = rng.choice(
        len(full),
        size=min(
            n,
            len(full),
        ),
        replace=False,
    )


    combos = []


    for i in idx:

        (
            tempo,
            mode,
            center,
            contour,
            motif,
            npb,
            rhythm,
            dynamics,
            articulation,
            instrument,
        ) = full[i]


        combos.append({
            "tempo_bpm": tempo,
            "mode": mode,
            "pitch_center_midi": center,
            "pitch_range": 12,
            "contour": contour,
            "motif_pattern": motif,
            "notes_per_beat": npb,
            "rhythm_pattern": rhythm,
            "dynamics": dynamics,
            "articulation": articulation,
            "instrument": instrument,
        })


    return combos


# ---------------------------------------------------------------------------
# WAV rendering
# ---------------------------------------------------------------------------

def render(
    p: dict,
    soundfont: str,
    out_wav: str,
    duration: float = 3.0,
    sr: int = 22050,
):
    """Render parameters to WAV via FluidSynth.

    Output is trimmed to exactly `duration`.

    FluidSynth can produce reverb/release tails after the MIDI notes finish,
    therefore the waveform is cut to the requested duration, faded briefly,
    and peak-normalised according to the requested dynamics parameter.
    """

    import soundfile as sf


    pm = params_to_midi(
        p,
        duration,
    )


    fd_m, mid = tempfile.mkstemp(
        suffix=".mid"
    )

    os.close(
        fd_m
    )


    fd_w, tmp_wav = tempfile.mkstemp(
        suffix=".wav"
    )

    os.close(
        fd_w
    )


    try:

        pm.write(
            mid
        )


        cmd = [
            "fluidsynth",
            "-ni",
            "-g",
            "1.0",
            "-r",
            str(sr),
            "-T",
            "wav",
            "-F",
            tmp_wav,
            soundfont,
            mid,
        ]


        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )


        if (
            r.returncode != 0
            or not os.path.exists(
                tmp_wav
            )
        ):

            raise RuntimeError(
                "fluidsynth failed: "
                f"{r.stderr.strip()[:300]}"
            )


        y, _ = sf.read(
            tmp_wav
        )


        if y.ndim > 1:

            y = y.mean(
                axis=1
            )


        n = int(
            duration
            * sr
        )


        if len(y) >= n:

            y = y[:n]

        else:

            y = np.pad(
                y,
                (
                    0,
                    n - len(y),
                ),
            )


        fade = min(
            int(
                0.06
                * sr
            ),
            len(y),
        )


        if fade > 0:

            y[-fade:] *= np.linspace(
                1.0,
                0.0,
                fade,
            )


        target = {
            "soft": 0.4,
            "moderate": 0.7,
            "loud": 0.95,
        }[
            p["dynamics"]
        ]


        peak = float(
            np.max(
                np.abs(
                    y
                )
            )
        ) or 1.0


        y = (
            y
            * (
                target
                / peak
            )
        ).astype(
            np.float32
        )


        sf.write(
            out_wav,
            y,
            sr,
        )


    finally:

        for f in (
            mid,
            tmp_wav,
        ):

            if os.path.exists(
                f
            ):

                os.unlink(
                    f
                )


    return out_wav