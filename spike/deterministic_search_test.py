"""Diagnostic comparison: current Qwen optimisation versus a fixed,
independent deterministic parameter pool.

This script does NOT modify the official stimuli, manifest, briefs, prompts,
or optimiser.

For each current pilot run:
  - candidate 0 is that run's Qwen-generated non-optimised logo;
  - the remaining candidates come from one fixed independent grid_params pool;
  - candidates are selected using the same quadrant-first / distance-second rule
    as the official optimisation controller.

The fixed pool uses a different seed from probe_reachable.py, so it is not the
same point sample that defined the reachable-region bounds.
"""

from __future__ import annotations

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
from generator.synth import render, grid_params


POOL_SEED = 20260829
BUDGETS = [10, 30, 100, 300]


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
    # Exact logic used by the official controller when
    # require_quadrant=True.
    if cand["quadrant_ok"] != cur["quadrant_ok"]:
        return cand["quadrant_ok"]

    return cand["distance"] < cur["distance"]


def score_params(params, predict, soundfont, duration, sr, wav_path):
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

    return predict(
        combined_features(
            y,
            actual_sr,
        )
    )


def choose_for_target(record, pool, budget):
    target = tuple(record["target"])

    initial = record["non_optimised"]

    best = {
        "source": "initial",
        "params": initial["params"],
        "est": tuple(initial["est"]),
        "distance": float(initial["distance"]),
        "quadrant_ok": bool(initial["quadrant_ok"]),
    }

    # Budget includes the initial candidate.
    for idx, item in enumerate(
        pool[: max(0, budget - 1)],
        start=1,
    ):
        est = item["est"]

        cand = {
            "source": f"pool_{idx}",
            "params": item["params"],
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

        if better(cand, best):
            best = cand

    return best


def summary(rows, threshold):
    n = len(rows)

    both = sum(
        r["distance"] <= threshold
        and r["quadrant_ok"]
        for r in rows
    )

    within = sum(
        r["distance"] <= threshold
        for r in rows
    )

    quad = sum(
        r["quadrant_ok"]
        for r in rows
    )

    mean_d = sum(
        r["distance"]
        for r in rows
    ) / n

    return {
        "n": n,
        "both": both,
        "within": within,
        "quad": quad,
        "mean_distance": mean_d,
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

    if len(manifest) != 12:
        print(
            f"WARNING: current manifest has {len(manifest)} runs, "
            "not the expected 12-run four-brief pilot."
        )

    exp = load("experiment.yaml")

    threshold = float(
        exp["generation"]["optimisation"]["threshold"]
    )

    soundfont = str(
        ROOT
        / exp["synthesis"]["soundfont"]["path"]
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
        r.get("coach")
        for r in manifest
        if r.get("coach")
    }

    if len(coaches) > 1:
        raise RuntimeError(
            f"Manifest contains multiple coaches: {coaches}"
        )

    coach = (
        next(iter(coaches))
        if coaches
        else "estimator_A2"
    )

    predict, meta = load_estimator(
        coach
    )

    print(
        f"coach: {coach} "
        f"({meta['corpus']}/{meta['model_family']})"
    )

    print(
        f"threshold: {threshold}"
    )

    print(
        f"soundfont: {soundfont}"
    )

    print(
        f"independent pool seed: {POOL_SEED}"
    )

    print(
        "reachable-region probe seed: 0"
    )

    # 300-budget comparison means:
    # 1 initial + 299 independent deterministic candidates.
    pool_params = grid_params(
        max(BUDGETS) - 1,
        seed=POOL_SEED,
    )

    print(
        f"\nScoring {len(pool_params)} independent "
        "deterministic candidates once through A2..."
    )

    pool = []

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(
            tmp,
            "candidate.wav",
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            for i, params in enumerate(
                pool_params,
                start=1,
            ):
                try:
                    est = score_params(
                        params,
                        predict,
                        soundfont,
                        duration,
                        sr,
                        wav_path,
                    )

                except Exception as exc:
                    print(
                        f"  candidate {i} FAILED: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                pool.append({
                    "params": params,
                    "est": (
                        float(est[0]),
                        float(est[1]),
                    ),
                })

                if i % 50 == 0:
                    print(
                        f"  {i}/{len(pool_params)}"
                    )

    print(
        f"\nSuccessfully scored "
        f"{len(pool)}/{len(pool_params)} candidates."
    )

    # Current Qwen results.
    qwen_rows = []

    for r in manifest:
        opt = r["optimised"]

        qwen_rows.append({
            "brief": r["brief"],
            "run": r["run"],
            "distance": float(
                opt["distance"]
            ),
            "quadrant_ok": bool(
                opt["quadrant_ok"]
            ),
        })

    deterministic = {}

    for budget in BUDGETS:
        deterministic[budget] = [
            {
                "brief": r["brief"],
                "run": r["run"],
                **choose_for_target(
                    r,
                    pool,
                    budget,
                ),
            }
            for r in manifest
        ]

    print(
        "\n"
        + "=" * 110
    )

    print(
        "PER-RUN COMPARISON"
    )

    print(
        "=" * 110
    )

    header = (
        f"{'brief':5s} "
        f"{'run':>3s} | "
        f"{'Qwen d':>7s} {'Q':>2s} | "
        f"{'Det10 d':>7s} {'Q':>2s} | "
        f"{'Det30 d':>7s} {'Q':>2s} | "
        f"{'Det100 d':>8s} {'Q':>2s} | "
        f"{'Det300 d':>8s} {'Q':>2s}"
    )

    print(header)
    print("-" * len(header))

    for i, r in enumerate(manifest):
        q = qwen_rows[i]

        values = [
            deterministic[b][i]
            for b in BUDGETS
        ]

        print(
            f"{r['brief']:5s} "
            f"{r['run']:3d} | "
            f"{q['distance']:7.3f} "
            f"{'Y' if q['quadrant_ok'] else 'N':>2s} | "
            f"{values[0]['distance']:7.3f} "
            f"{'Y' if values[0]['quadrant_ok'] else 'N':>2s} | "
            f"{values[1]['distance']:7.3f} "
            f"{'Y' if values[1]['quadrant_ok'] else 'N':>2s} | "
            f"{values[2]['distance']:8.3f} "
            f"{'Y' if values[2]['quadrant_ok'] else 'N':>2s} | "
            f"{values[3]['distance']:8.3f} "
            f"{'Y' if values[3]['quadrant_ok'] else 'N':>2s}"
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 90
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
        s = summary(
            deterministic[budget],
            threshold,
        )

        print(
            f"Det-{budget:<7d}: "
            f"mean d={s['mean_distance']:.3f} | "
            f"<=threshold={s['within']}/{s['n']} | "
            f"quadrant={s['quad']}/{s['n']} | "
            f"BOTH={s['both']}/{s['n']}"
        )

    print(
        "\nBest deterministic candidates at budget 300:"
    )

    for row in deterministic[300]:
        p = row["params"]

        print(
            f"{row['brief']} run{row['run']} "
            f"d={row['distance']:.3f} "
            f"quad={row['quadrant_ok']} "
            f"source={row['source']} | "
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


if __name__ == "__main__":
    main()
