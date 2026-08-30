"""Build the target-blind acoustic calibration reference used by audience agents.

The calibration set is independent of the 96 study stimuli. It samples the same
parametric synthesizer space with grid_params(), renders 300 short sounds, extracts
only the audience acoustic descriptors, and stores their distributions.

No brand brief, target VA coordinate, condition, estimator output or emotion label
is used.

Run from project root:
    python -W ignore src/audience/build_feature_reference.py
"""

import sys
import json
import argparse
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_loader import load, ROOT
from generator.synth import grid_params, render
from features.extracts import (
    load_audio,
    extract_audience_block,
    build_audience_reference,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soundfont", default=None)
    ap.add_argument(
        "--out",
        default="models/audience_feature_reference.json"
    )

    args = ap.parse_args()

    exp = load("experiment.yaml")

    sr = int(exp["synthesis"]["sample_rate_hz"])
    duration = float(exp["synthesis"]["duration_s"])

    soundfont = (
        args.soundfont
        or str(ROOT / exp["synthesis"]["soundfont"]["path"])
    )

    params = grid_params(args.n, seed=args.seed)

    blocks = []

    print(
        f"building target-blind audience feature reference "
        f"from {len(params)} synthetic sounds..."
    )

    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, p in enumerate(params, 1):

            wav = str(Path(tmp) / "reference.wav")

            try:
                render(
                    p,
                    soundfont,
                    wav,
                    duration,
                    sr
                )

                y, _sr = load_audio(
                    wav,
                    sr=sr
                )

                blocks.append(
                    extract_audience_block(y, _sr)
                )

            except Exception as e:
                print(
                    f"  skipped reference sound {i}: {e}"
                )

            if i % 50 == 0:
                print(
                    f"  {i}/{len(params)}"
                )

    if len(blocks) < max(50, int(0.8 * len(params))):
        raise SystemExit(
            f"only {len(blocks)}/{len(params)} "
            f"reference sounds succeeded; refusing to continue"
        )

    reference = build_audience_reference(blocks)

    reference["basis"] = (
        "independent target-blind synthetic calibration set "
        "from generator parameter space"
    )

    reference["n_requested"] = len(params)
    reference["n_successful"] = len(blocks)
    reference["grid_seed"] = args.seed
    reference["duration_s"] = duration
    reference["sample_rate_hz"] = sr

    reference["excludes"] = [
        "brand_brief",
        "target_coordinates",
        "condition",
        "estimator_outputs",
        "emotion_labels",
    ]

    out = ROOT / args.out

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            reference,
            indent=2,
            sort_keys=True
        )
    )

    print(
        f"saved {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()