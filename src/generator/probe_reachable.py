"""Map the reachable valence-arousal region for the current synth + Estimator A.

Renders a spread of logos spanning the parameter space, scores each with the frozen
Estimator A, and reports where the estimates actually land. That point cloud is the
region briefs can realistically target; targets outside it are unreachable.

    python src/generator/probe_reachable.py            # config soundfont
    python src/generator/probe_reachable.py --soundfont assets/soundfonts/GeneralUser-GS.sf2
"""
import sys
import os
import json
import glob
import argparse
import itertools
import warnings
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT                       # noqa: E402
from estimators.model import load as load_estimator         # noqa: E402
from estimators.data import combined_features               # noqa: E402
from features.extracts import load_audio                     # noqa: E402
from generator.synth import render, grid_params             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundfont")
    ap.add_argument("--n", type=int, default=300, help="number of logos to probe")
    ap.add_argument("--coach", default="estimator_A",
                    help="Estimator that defines the reachable region. Must match the "
                         "coach used for generation.")
    args = ap.parse_args()

    exp = load("experiment.yaml")
    sf = args.soundfont or str(ROOT / exp["synthesis"]["soundfont"]["path"])
    duration = exp.get("synthesis", {}).get("duration_s", 3.0)
    sr = exp.get("synthesis", {}).get("sample_rate_hz", 22050)
    predict, meta = load_estimator(args.coach)

    combos = grid_params(args.n)
    print(f"probing {len(combos)} logos through the synth + {args.coach} "
          f"({meta['corpus']}/{meta['model_family']})...")
    ests = []
    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, p in enumerate(combos, 1):
                wav = os.path.join(tmp, "probe.wav")
                try:
                    render(p, sf, wav, duration, sr)
                    y, _sr = load_audio(wav, sr=sr)
                    v, a = predict(combined_features(y, _sr))
                    ests.append((v, a))
                except Exception:
                    continue
                if i % 50 == 0:
                    print(f"  {i}/{len(combos)}")

    ests = np.array(ests)
    v, a = ests[:, 0], ests[:, 1]
    report = {
        "coach": args.coach,
        "n": len(ests),
        "valence": {"min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
                    "p5": round(float(np.percentile(v, 5)), 3),
                    "p95": round(float(np.percentile(v, 95)), 3),
                    "mean": round(float(v.mean()), 3)},
        "arousal": {"min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                    "p5": round(float(np.percentile(a, 5)), 3),
                    "p95": round(float(np.percentile(a, 95)), 3),
                    "mean": round(float(a.mean()), 3)},
    }
    out = ROOT / "models" / "reachable_va.json"
    out.write_text(json.dumps({"report": report,
                               "estimates": [[round(x, 3), round(y, 3)] for x, y in ests]}, indent=2))

    print(f"\n=== Reachable VA region ({args.coach} on synthetic logos) ===")
    print(f"  valence:  {report['valence']['min']:+.2f} .. {report['valence']['max']:+.2f}   "
          f"(p5..p95: {report['valence']['p5']:+.2f} .. {report['valence']['p95']:+.2f})")
    print(f"  arousal:  {report['arousal']['min']:+.2f} .. {report['arousal']['max']:+.2f}   "
          f"(p5..p95: {report['arousal']['p5']:+.2f} .. {report['arousal']['p95']:+.2f})")
    print(f"\nsaved {out.relative_to(ROOT)} (region + point cloud)")
    print("Use the p5..p95 span as the target extent; generate_briefs.py reads this file.")


if __name__ == "__main__":
    main()