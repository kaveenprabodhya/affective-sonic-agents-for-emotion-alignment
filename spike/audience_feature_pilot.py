"""Small validation gate for the calibrated acoustic-feature -> Qwen interface.

Tests only the neutral and generic controls on acoustically extreme study
stimuli. Qwen never receives target coordinates, intended quadrants,
condition labels, brand briefs or estimator outputs.

Run:
    python -W ignore spike/audience_feature_pilot.py \
        --backend ollama \
        --model qwen3:8b \
        --reps 3
"""

import sys
import json
import argparse
import statistics
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from config_loader import load, ROOT, LOGS
from llm.client import LLMClient, call_seed
from audience.survey import run_survey
from features.extracts import (
    load_audio,
    extract_audience_block,
    format_audience_block,
    AUDIENCE_REFERENCE_KEYS,
)


def build_stimuli(manifest):
    """Convert manifest pairs into the 96 individual stimuli."""
    stimuli = []

    for r in manifest:
        for condition, key in (
            ("non_optimised", "non_optimised"),
            ("optimised", "optimised"),
        ):
            stimuli.append(
                {
                    "file": r[condition]["file"],
                    "brief": r["brief"],
                    "condition": key,
                    "target": r["target"],
                }
            )

    return stimuli


def choose_extremes(stimuli, max_stimuli=14):
    """Select unique min/max examples across acoustic dimensions."""
    chosen = []
    seen = set()

    for key in AUDIENCE_REFERENCE_KEYS:

        valid = [
            s
            for s in stimuli
            if s["fblock"].get(key) is not None
        ]

        if not valid:
            continue

        ordered = sorted(
            valid,
            key=lambda s: float(s["fblock"][key]),
        )

        for s in (ordered[0], ordered[-1]):

            if s["file"] not in seen:
                seen.add(s["file"])
                chosen.append(s)

    return chosen[:max_stimuli]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "anthropic", "mock"],
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--host",
        default="http://localhost:11434",
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-stimuli",
        type=int,
        default=14,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    exp = load("experiment.yaml")

    sr = int(
        exp["synthesis"]["sample_rate_hz"]
    )

    manifest_path = (
        ROOT
        / "data"
        / "stimuli"
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise SystemExit(
            "Missing data/stimuli/manifest.json"
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    stimuli = build_stimuli(
        manifest
    )

    # ---------------------------------------------------------
    # Load EXACT current audience prompts
    # ---------------------------------------------------------

    P = ROOT / "config" / "prompts"

    cfg = {
        "questionnaire": load(
            "questionnaire.yaml"
        ),
        "personas": load(
            "personas.yaml"
        ),
        "prompts": {
            "audience_system":
                (P / "audience_system.txt").read_text(),

            "audience_system_neutral":
                (P / "audience_system_neutral.txt").read_text(),

            "audience_system_generic":
                (P / "audience_system_generic.txt").read_text(),

            "audience_user":
                (P / "audience_user.txt").read_text(),
        },
    }

    # ---------------------------------------------------------
    # Load independent acoustic calibration reference
    # ---------------------------------------------------------

    reference_path = (
        ROOT
        / "models"
        / "audience_feature_reference.json"
    )

    if not reference_path.exists():

        raise SystemExit(
            "\nMissing models/audience_feature_reference.json.\n\n"
            "Run this first:\n\n"
            "python -W ignore "
            "src/audience/build_feature_reference.py\n"
        )

    reference = json.loads(
        reference_path.read_text()
    )

    # ---------------------------------------------------------
    # Extract features from the 96 fixed study stimuli
    # ---------------------------------------------------------

    stim_dir = (
        ROOT
        / "data"
        / "stimuli"
    )

    print(
        f"\nExtracting acoustic features from "
        f"{len(stimuli)} study stimuli..."
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        for i, stimulus in enumerate(
            stimuli,
            1,
        ):

            wav_path = (
                stim_dir
                / stimulus["file"]
            )

            if not wav_path.exists():
                raise SystemExit(
                    f"Missing stimulus: {wav_path}"
                )

            y, actual_sr = load_audio(
                str(wav_path),
                sr=sr,
            )

            stimulus["fblock"] = (
                extract_audience_block(
                    y,
                    actual_sr,
                )
            )

            stimulus["ftext"] = (
                format_audience_block(
                    stimulus["fblock"],
                    reference,
                )
            )

            if i % 20 == 0:
                print(
                    f"  extracted {i}/{len(stimuli)}"
                )

    # ---------------------------------------------------------
    # Select acoustically extreme stimuli
    # ---------------------------------------------------------

    selected = stimuli

    print(
        f"\nTesting all {len(selected)} study stimuli."
    )

    print(
        "Testing only NEUTRAL and GENERIC controls."
    )

    print(
        "No intended VA coordinates are shown to Qwen.\n"
    )

    # ---------------------------------------------------------
    # Qwen
    # ---------------------------------------------------------

    model_cfg = (
        exp["models"]["audience_primary"]
    )

    model = (
        args.model
        or model_cfg["checkpoint"]
    )

    if (
        args.backend == "ollama"
        and model in (
            None,
            "TBD_at_pilot",
        )
    ):
        model = "qwen3:8b"

    temperature = float(
        model_cfg.get(
            "temperature",
            0.7,
        )
    )

    think = bool(
        model_cfg.get(
            "think",
            False,
        )
    )

    LOGS.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = LLMClient(
        backend=args.backend,
        model=model,
        temperature=temperature,
        think=think,
        host=args.host,
        log_path=str(
            LOGS
            / "audience_feature_pilot.jsonl"
        ),
    )

    print(
        f"backend={args.backend}"
    )

    print(
        f"model={model}"
    )

    print(
        f"temperature={temperature}"
    )

    print(
        f"think={think}\n"
    )

    # ---------------------------------------------------------
    # Run controls
    # ---------------------------------------------------------

    rows = []

    total_calls = (
        2
        * len(selected)
        * args.reps
    )

    current = 0

    for kind in (
        "neutral",
        "generic",
    ):

        for stimulus in selected:

            for rep in range(
                args.reps
            ):

                current += 1

                obj, err, _ = run_survey(
                    client,
                    None,
                    stimulus["ftext"],
                    cfg,
                    retries=exp["audience"]["retries"],
                    agent_kind=kind,
                    seed=call_seed(
                        "audience_feature_pilot",
                        kind,
                        stimulus["file"],
                        rep,
                    ),
                )

                if obj is None:

                    print(
                        f"INVALID "
                        f"{kind} "
                        f"{stimulus['file']} "
                        f"rep={rep}: "
                        f"{err}"
                    )

                    continue

                rows.append(
                    {
                        "kind": kind,
                        "file": stimulus["file"],
                        "Q1": obj["Q1"],
                        "Q2": obj["Q2"],
                    }
                )

                print(
                    f"[{current:>3}/{total_calls}] "
                    f"{kind:<7} "
                    f"{stimulus['file']:<28} "
                    f"Q1={obj['Q1']} "
                    f"Q2={obj['Q2']}"
                )

    if not rows:

        raise SystemExit(
            "No valid pilot responses."
        )

    # ---------------------------------------------------------
    # SCALE DIAGNOSTIC
    # ---------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "SCALE-USE DIAGNOSTIC"
    )

    print(
        "========================================"
    )

    for q in (
        "Q1",
        "Q2",
    ):

        vals = [
            r[q]
            for r in rows
        ]

        counts = Counter(
            vals
        )

        below = (
            100
            * sum(
                v < 5
                for v in vals
            )
            / len(vals)
        )

        midpoint = (
            100
            * sum(
                v == 5
                for v in vals
            )
            / len(vals)
        )

        above = (
            100
            * sum(
                v > 5
                for v in vals
            )
            / len(vals)
        )

        print(
            f"\n{q}"
        )

        print(
            f"  mean: "
            f"{statistics.mean(vals):.2f}"
        )

        print(
            f"  values used: "
            f"{sorted(counts)}"
        )

        print(
            f"  below midpoint (<5): "
            f"{below:.1f}%"
        )

        print(
            f"  midpoint (=5): "
            f"{midpoint:.1f}%"
        )

        print(
            f"  above midpoint (>5): "
            f"{above:.1f}%"
        )

        print(
            "  counts: "
            + " ".join(
                f"{value}:{counts[value]}"
                for value
                in sorted(counts)
            )
        )

    # ---------------------------------------------------------
    # STIMULUS DIFFERENTIATION
    # ---------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "MEAN RESPONSE BY STIMULUS"
    )

    print(
        "========================================\n"
    )

    stimulus_means = []

    for stimulus in selected:

        cells = [
            r
            for r in rows
            if r["file"]
            == stimulus["file"]
        ]

        if not cells:
            continue

        q1_mean = statistics.mean(
            r["Q1"]
            for r in cells
        )

        q2_mean = statistics.mean(
            r["Q2"]
            for r in cells
        )

        stimulus_means.append(
            (
                stimulus["file"],
                q1_mean,
                q2_mean,
            )
        )

        print(
            f"{stimulus['file']:<30} "
            f"Q1={q1_mean:.2f}   "
            f"Q2={q2_mean:.2f}"
        )

    q1_stim_means = [
        x[1]
        for x in stimulus_means
    ]

    q2_stim_means = [
        x[2]
        for x in stimulus_means
    ]

    print(
        "\n"
        "========================================"
    )

    print(
        "STIMULUS RANGE"
    )

    print(
        "========================================"
    )

    if q1_stim_means:

        print(
            f"Q1 stimulus-mean range: "
            f"{min(q1_stim_means):.2f} "
            f"to "
            f"{max(q1_stim_means):.2f}"
        )

    if q2_stim_means:

        print(
            f"Q2 stimulus-mean range: "
            f"{min(q2_stim_means):.2f} "
            f"to "
            f"{max(q2_stim_means):.2f}"
        )

    # ---------------------------------------------------------
    # Final interpretation
    # ---------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "HOW TO READ THIS"
    )

    print(
        "========================================"
    )

    q1_vals = [
    r["Q1"]
    for r in rows
    ]

    q2_vals = [
        r["Q2"]
        for r in rows
    ]

    q1_below = sum(
        v < 5
        for v in q1_vals
    )

    q2_below = sum(
        v < 5
        for v in q2_vals
    )

    q1_above = sum(
        v > 5
        for v in q1_vals
    )

    q2_above = sum(
        v > 5
        for v in q2_vals
    )

    q1_range = (
        max(q1_stim_means)
        - min(q1_stim_means)
    )

    q2_range = (
        max(q2_stim_means)
        - min(q2_stim_means)
    )

    print(
        f"\nQ1 responses below midpoint: "
        f"{q1_below}/{len(q1_vals)}"
    )

    print(
        f"Q1 responses above midpoint: "
        f"{q1_above}/{len(q1_vals)}"
    )

    print(
        f"Q2 responses below midpoint: "
        f"{q2_below}/{len(q2_vals)}"
    )

    print(
        f"Q2 responses above midpoint: "
        f"{q2_above}/{len(q2_vals)}"
    )

    print(
        f"Q1 stimulus differentiation range: "
        f"{q1_range:.2f}"
    )

    print(
        f"Q2 stimulus differentiation range: "
        f"{q2_range:.2f}"
    )

    q1_has_low_stimulus = min(q1_stim_means) < 5
    q1_has_high_stimulus = max(q1_stim_means) > 5

    q2_has_low_stimulus = min(q2_stim_means) < 5
    q2_has_high_stimulus = max(q2_stim_means) > 5


    if not (
        q1_has_low_stimulus
        and q1_has_high_stimulus
        and q2_has_low_stimulus
        and q2_has_high_stimulus
    ):

        print(
            "\nRESULT: STIMULUS-LEVEL QUADRANT "
            "DISCRIMINATION IS STILL INCOMPLETE."
        )

        print(
            f"Q1 crosses midpoint: "
            f"{q1_has_low_stimulus and q1_has_high_stimulus}"
        )

        print(
            f"Q2 crosses midpoint: "
            f"{q2_has_low_stimulus and q2_has_high_stimulus}"
        )

    elif q1_range < 1.0 or q2_range < 1.0:

        print(
            "\nRESULT: TWO-SIDED SCALE USE EXISTS, "
            "BUT STIMULUS SEPARATION IS WEAK."
        )

    else:

        print(
            "\nRESULT: PILOT PASSED. "
            "BOTH VA AXES CROSS THE MIDPOINT AT "
            "THE STIMULUS LEVEL AND SHOW "
            "MEANINGFUL STIMULUS DIFFERENTIATION."
        )


if __name__ == "__main__":
    main()
