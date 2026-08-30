"""Diagnostic local deterministic optimisation test.

Purpose
-------
Test whether a deterministic local parameter search can optimise the existing
Qwen-generated initial sonic-logo designs more effectively than Qwen revisions.

This script:
- reads the current 12-run B01/B05/B09/B13 manifest;
- treats each non-optimised Qwen design as the starting point;
- changes parameters locally;
- scores every candidate with the same frozen coach used by generation;
- uses the same quadrant-first, distance-second selection rule;
- compares budgets of 10, 20, 30 and 60 total evaluated candidates;
- writes NOTHING to data/stimuli or the official manifest.

Search behaviour
----------------
1. Start from the Qwen-generated non-optimised parameter set.
2. Explore one-parameter neighbours.
3. If all one-parameter neighbours around the current best have already been
   tested, explore two-parameter neighbours.
4. Retain a candidate only when it beats the current best using the same
   quadrant-first / distance-second rule as the official controller.
5. Candidate ordering is deterministic but run-specific.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config_loader import load
from estimators.model import load as load_estimator
from estimators.data import combined_features
from features.extracts import load_audio
from generator.synth import render, SCHEMA


BUDGETS = [10, 20, 30, 60]
MAX_BUDGET = max(BUDGETS)

# Local steps for continuous/integer parameters.
INT_STEPS = {
    "tempo_bpm": 20,
    "pitch_center_midi": 4,
    "pitch_range": 4,
}


def param_key(params):
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
    )


def dist(est, target):
    return math.hypot(
        est[0] - target[0],
        est[1] - target[1],
    )


def quadrant_ok(est, target):
    for e, t in zip(est, target):
        if t == 0:
            continue

        if e * t <= 0:
            return False

    return True


def better(cand, cur):
    """Same quadrant-first / distance-second logic as generator.loop."""

    if cand["quadrant_ok"] != cur["quadrant_ok"]:
        return cand["quadrant_ok"]

    return cand["distance"] < cur["distance"]


def alternatives(params):
    """Legal local alternatives for every parameter.

    Integer parameters move one local step in either direction.
    Enumerated parameters may switch to any other legal category.
    """

    out = {}

    for name, spec in SCHEMA.items():

        current = params[name]

        if spec[0] == "int":
            lo = int(spec[1])
            hi = int(spec[2])
            step = INT_STEPS[name]

            values = []

            for value in (
                current - step,
                current + step,
            ):
                value = max(
                    lo,
                    min(
                        hi,
                        int(value),
                    ),
                )

                if (
                    value != current
                    and value not in values
                ):
                    values.append(value)

            out[name] = values

        else:
            out[name] = [
                value
                for value in spec[1]
                if value != current
            ]

    return out


def single_neighbours(params):
    """Candidates differing from current best on exactly one parameter."""

    alts = alternatives(params)

    candidates = []

    for name, values in alts.items():
        for value in values:

            cand = dict(params)
            cand[name] = value

            candidates.append(cand)

    return candidates


def pair_neighbours(params):
    """Candidates differing on exactly two parameter dimensions.

    These are used only after unseen one-parameter neighbours around the
    current best have been exhausted.
    """

    alts = alternatives(params)

    active_keys = [
        key
        for key, values in alts.items()
        if values
    ]

    candidates = []

    for key1, key2 in itertools.combinations(
        active_keys,
        2,
    ):
        for value1 in alts[key1]:
            for value2 in alts[key2]:

                cand = dict(params)

                cand[key1] = value1
                cand[key2] = value2

                candidates.append(cand)

    return candidates


def deterministic_order(
    candidates,
    brief_id,
    run_idx,
    best_params,
    layer,
):
    """Stable, reproducible but run-specific candidate ordering."""

    salt = (
        f"{brief_id}|{run_idx}|{layer}|"
        f"{param_key(best_params)}"
    )

    def rank(candidate):
        text = (
            salt
            + "|"
            + param_key(candidate)
        )

        return hashlib.sha256(
            text.encode()
        ).hexdigest()

    return sorted(
        candidates,
        key=rank,
    )


def score_params(
    params,
    predict,
    soundfont,
    duration,
    sr,
    wav_path,
):
    render(
        params,
        soundfont,
        wav_path,
        duration,
        sr,
    )

    y, actual_sr = load_audio(
        wav_path,
        sr=sr,
    )

    est = predict(
        combined_features(
            y,
            actual_sr,
        )
    )

    return (
        float(est[0]),
        float(est[1]),
    )


def changed_dimensions(initial, final):
    return sum(
        initial[k] != final[k]
        for k in initial
    )


def snapshot(best):
    return {
        "distance": float(
            best["distance"]
        ),
        "quadrant_ok": bool(
            best["quadrant_ok"]
        ),
        "params": dict(
            best["params"]
        ),
        "est": tuple(
            best["est"]
        ),
    }


def run_local_search(
    record,
    predict,
    soundfont,
    duration,
    sr,
    threshold,
    score_cache,
    wav_path,
):
    """Run one local search up to MAX_BUDGET.

    Budget includes the original non-optimised Qwen candidate as evaluation 1.
    """

    target = tuple(
        float(x)
        for x in record["target"]
    )

    initial = record["non_optimised"]

    initial_params = dict(
        initial["params"]
    )

    initial_est = tuple(
        float(x)
        for x in initial["est"]
    )

    best = {
        "params": initial_params,
        "est": initial_est,
        "distance": float(
            initial["distance"]
        ),
        "quadrant_ok": bool(
            initial["quadrant_ok"]
        ),
    }

    # The initial Qwen candidate counts as evaluation 1.
    evaluations = 1

    seen = {
        param_key(
            initial_params
        )
    }

    # Cache its A2 result too.
    score_cache.setdefault(
        param_key(initial_params),
        initial_est,
    )

    checkpoints = {}

    if evaluations in BUDGETS:
        checkpoints[evaluations] = snapshot(
            best
        )

    while evaluations < MAX_BUDGET:

        # ------------------------------------------------------------
        # First search unseen one-parameter neighbours of current best.
        # ------------------------------------------------------------
        candidates = [
            cand
            for cand in single_neighbours(
                best["params"]
            )
            if param_key(cand) not in seen
        ]

        layer = "single"

        # ------------------------------------------------------------
        # If those are exhausted, allow two-parameter local moves.
        # This lets the search escape a coordinate-wise local optimum.
        # ------------------------------------------------------------
        if not candidates:

            candidates = [
                cand
                for cand in pair_neighbours(
                    best["params"]
                )
                if param_key(cand) not in seen
            ]

            layer = "pair"

        if not candidates:
            break

        candidates = deterministic_order(
            candidates,
            record["brief"],
            record["run"],
            best["params"],
            layer,
        )

        candidate_params = candidates[0]
        key = param_key(
            candidate_params
        )

        seen.add(key)

        if key in score_cache:
            est = score_cache[key]

        else:
            try:
                est = score_params(
                    candidate_params,
                    predict,
                    soundfont,
                    duration,
                    sr,
                    wav_path,
                )

            except Exception as exc:
                print(
                    f"WARNING "
                    f"{record['brief']} "
                    f"run{record['run']} "
                    f"candidate failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                continue

            score_cache[key] = est

        evaluations += 1

        cand = {
            "params": candidate_params,
            "est": est,
            "distance": dist(
                est,
                target,
            ),
            "quadrant_ok": quadrant_ok(
                est,
                target,
            ),
        }

        if better(
            cand,
            best,
        ):
            best = cand

        if evaluations in BUDGETS:
            checkpoints[evaluations] = snapshot(
                best
            )

        # Same success rule as the official pipeline.
        if (
            best["distance"] <= threshold
            and best["quadrant_ok"]
        ):
            break

    # If the search stopped before one or more requested budgets,
    # later budgets inherit the final best result.
    final = snapshot(
        best
    )

    for budget in BUDGETS:

        if budget not in checkpoints:

            earlier = [
                b
                for b in checkpoints
                if b <= budget
            ]

            if earlier:
                checkpoints[budget] = dict(
                    checkpoints[max(earlier)]
                )
            else:
                checkpoints[budget] = dict(
                    final
                )

    # If success happened before later budgets, use that successful best
    # for every later checkpoint.
    last = final

    for budget in BUDGETS:
        if budget > evaluations:
            checkpoints[budget] = dict(
                last
            )

    return {
        "initial_params": initial_params,
        "evaluations_used": evaluations,
        "checkpoints": checkpoints,
        "final": final,
    }


def summary(rows, threshold):
    n = len(rows)

    within = sum(
        row["distance"] <= threshold
        for row in rows
    )

    quad = sum(
        row["quadrant_ok"]
        for row in rows
    )

    both = sum(
        row["distance"] <= threshold
        and row["quadrant_ok"]
        for row in rows
    )

    mean_distance = sum(
        row["distance"]
        for row in rows
    ) / n

    return {
        "n": n,
        "within": within,
        "quad": quad,
        "both": both,
        "mean_distance": mean_distance,
    }


def main():

    manifest_path = (
        ROOT
        / "data"
        / "stimuli"
        / "manifest.json"
    )

    manifest = json.load(
        open(manifest_path)
    )

    print(
        f"current manifest: "
        f"{len(manifest)} runs"
    )

    expected = {
        "B01",
        "B05",
        "B09",
        "B13",
    }

    present = {
        row["brief"]
        for row in manifest
    }

    if len(manifest) != 12 or present != expected:
        raise RuntimeError(
            "This spike expects the current "
            "12-run B01/B05/B09/B13 pilot manifest. "
            f"Found {len(manifest)} runs and briefs "
            f"{sorted(present)}."
        )

    exp = load(
        "experiment.yaml"
    )

    threshold = float(
        exp["generation"][
            "optimisation"
        ]["threshold"]
    )

    soundfont = str(
        ROOT
        / exp["synthesis"][
            "soundfont"
        ]["path"]
    )

    duration = exp.get(
        "synthesis",
        {},
    ).get(
        "duration_s",
        3.0,
    )

    sr = exp.get(
        "synthesis",
        {},
    ).get(
        "sample_rate_hz",
        22050,
    )

    coaches = {
        row.get("coach")
        for row in manifest
        if row.get("coach")
    }

    if len(coaches) != 1:
        raise RuntimeError(
            f"Expected one coach in manifest; "
            f"found {coaches}"
        )

    coach = next(
        iter(coaches)
    )

    predict, meta = load_estimator(
        coach
    )

    print(
        f"coach: {coach} "
        f"({meta['corpus']}/"
        f"{meta['model_family']})"
    )

    print(
        f"threshold: {threshold}"
    )

    print(
        f"soundfont: {soundfont}"
    )

    print(
        "search: deterministic local "
        "single-parameter moves, then "
        "two-parameter escape moves"
    )

    print(
        f"budgets: {BUDGETS}"
    )

    # A2 score is a deterministic function of a complete parameter vector,
    # so identical vectors can safely share one cached estimator result.
    score_cache = {}

    local_results = []

    with tempfile.TemporaryDirectory() as tmp:

        wav_path = os.path.join(
            tmp,
            "candidate.wav",
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            for index, record in enumerate(
                manifest,
                start=1,
            ):

                print(
                    f"searching "
                    f"{record['brief']} "
                    f"run{record['run']} "
                    f"({index}/{len(manifest)})..."
                )

                result = run_local_search(
                    record,
                    predict,
                    soundfont,
                    duration,
                    sr,
                    threshold,
                    score_cache,
                    wav_path,
                )

                local_results.append(
                    result
                )

    # ------------------------------------------------------------
    # Current Qwen optimiser results.
    # ------------------------------------------------------------

    qwen_rows = [
        {
            "distance": float(
                row["optimised"][
                    "distance"
                ]
            ),
            "quadrant_ok": bool(
                row["optimised"][
                    "quadrant_ok"
                ]
            ),
        }
        for row in manifest
    ]

    print(
        "\n"
        + "=" * 120
    )

    print(
        "PER-RUN COMPARISON"
    )

    print(
        "=" * 120
    )

    header = (
        f"{'brief':5s} "
        f"{'run':>3s} | "
        f"{'Qwen d':>7s} {'Q':>2s} | "
        f"{'Loc10 d':>7s} {'Q':>2s} | "
        f"{'Loc20 d':>7s} {'Q':>2s} | "
        f"{'Loc30 d':>7s} {'Q':>2s} | "
        f"{'Loc60 d':>7s} {'Q':>2s} | "
        f"{'used':>4s}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for i, record in enumerate(
        manifest
    ):

        q = qwen_rows[i]

        local = local_results[i]

        c10 = local[
            "checkpoints"
        ][10]

        c20 = local[
            "checkpoints"
        ][20]

        c30 = local[
            "checkpoints"
        ][30]

        c60 = local[
            "checkpoints"
        ][60]

        print(
            f"{record['brief']:5s} "
            f"{record['run']:3d} | "
            f"{q['distance']:7.3f} "
            f"{'Y' if q['quadrant_ok'] else 'N':>2s} | "
            f"{c10['distance']:7.3f} "
            f"{'Y' if c10['quadrant_ok'] else 'N':>2s} | "
            f"{c20['distance']:7.3f} "
            f"{'Y' if c20['quadrant_ok'] else 'N':>2s} | "
            f"{c30['distance']:7.3f} "
            f"{'Y' if c30['quadrant_ok'] else 'N':>2s} | "
            f"{c60['distance']:7.3f} "
            f"{'Y' if c60['quadrant_ok'] else 'N':>2s} | "
            f"{local['evaluations_used']:4d}"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 100
    )

    qsum = summary(
        qwen_rows,
        threshold,
    )

    print(
        f"Qwen current : "
        f"mean d={qsum['mean_distance']:.3f} | "
        f"<=threshold={qsum['within']}/{qsum['n']} | "
        f"quadrant={qsum['quad']}/{qsum['n']} | "
        f"BOTH={qsum['both']}/{qsum['n']}"
    )

    for budget in BUDGETS:

        rows = [
            result[
                "checkpoints"
            ][budget]
            for result in local_results
        ]

        s = summary(
            rows,
            threshold,
        )

        print(
            f"Local-{budget:<3d}    : "
            f"mean d={s['mean_distance']:.3f} | "
            f"<=threshold={s['within']}/{s['n']} | "
            f"quadrant={s['quad']}/{s['n']} | "
            f"BOTH={s['both']}/{s['n']}"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL LOCAL CANDIDATES"
    )

    print(
        "=" * 100
    )

    for record, result in zip(
        manifest,
        local_results,
    ):

        final = result["final"]

        initial_params = result[
            "initial_params"
        ]

        p = final["params"]

        changed = changed_dimensions(
            initial_params,
            p,
        )

        print(
            f"{record['brief']} "
            f"run{record['run']} "
            f"d={final['distance']:.3f} "
            f"quad={final['quadrant_ok']} "
            f"evals={result['evaluations_used']} "
            f"changed_dims={changed} | "
            f"tempo={p['tempo_bpm']} "
            f"mode={p['mode']} "
            f"centre={p['pitch_center_midi']} "
            f"range={p['pitch_range']} "
            f"contour={p['contour']} "
            f"density={p['notes_per_beat']} "
            f"rhythm={p['rhythm_pattern']} "
            f"dyn={p['dynamics']} "
            f"art={p['articulation']} "
            f"inst={p['instrument']}"
        )

    print(
        "\nNo official files were modified."
    )


if __name__ == "__main__":
    main()
