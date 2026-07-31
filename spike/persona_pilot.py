"""Stage 2 gate: persona-differentiation pilot (mid-range stimuli + baselines).

Confirms, off the scale ceiling:
  1. valid completion,
  2. persona differentiation -- OCEAN personas differ on the SAME stimulus,
  3. feature tracking -- brighter/faster reads as higher arousal,
  4. baseline behaviour -- where the neutral and generic-listener baselines sit
     relative to the OCEAN spread (is the variation trait-driven, not framing or
     raw-model artefact?).

Stimuli are moderate (non-saturating) so personality has headroom to show.

    python spike/persona_pilot.py --backend ollama --model qwen3:8b
    python spike/persona_pilot.py --backend mock
"""
import sys
import time
import argparse
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_loader import load, ROOT, LOGS          # noqa: E402
from llm.client import LLMClient                      # noqa: E402
from audience.survey import run_survey                # noqa: E402
from features.extracts import format_audience_block    # noqa: E402

# Genuinely mid-range feature blocks: off both the scale ends AND the exact midpoint,
# so persona differences have the most headroom to show.
BLOCKS = {
    "moderate_bright": {"duration_s": 3.0, "tempo_bpm": 100.0, "mode": "major",
        "mean_pitch_midi": 64.0, "pitch_slope_midi_per_s": 0.15, "spectral_centroid_hz": 1750.0,
        "rms_energy": 0.07, "onset_rate_per_s": 3.2, "dynamic_range_db": 7.5},
    "moderate_dark": {"duration_s": 3.0, "tempo_bpm": 78.0, "mode": "minor",
        "mean_pitch_midi": 56.0, "pitch_slope_midi_per_s": -0.15, "spectral_centroid_hz": 1350.0,
        "rms_energy": 0.06, "onset_rate_per_s": 2.0, "dynamic_range_db": 6.0},
    "neutral_mid": {"duration_s": 3.0, "tempo_bpm": 90.0, "mode": "major",
        "mean_pitch_midi": 60.0, "pitch_slope_midi_per_s": 0.0, "spectral_centroid_hz": 1550.0,
        "rms_energy": 0.065, "onset_rate_per_s": 2.6, "dynamic_range_db": 6.5},
}


def pick_personas(personas):
    by = {p["id"]: p for p in personas}

    def find(**kw):
        for p in personas:
            if all(p[k] == v for k, v in kw.items()):
                return p["id"]

    ids = ["P01", "P32",
           find(extraversion="high", neuroticism="low"),
           find(extraversion="low", neuroticism="high"),
           find(openness="high", extraversion="high", neuroticism="high"),
           find(agreeableness="low", extraversion="low", neuroticism="low")]
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(by[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    exp = load("experiment.yaml")
    P = ROOT / "config/prompts"
    cfg = {"questionnaire": load("questionnaire.yaml"),
           "personas": load("personas.yaml"),
           "prompts": {"audience_system": (P / "audience_system.txt").read_text(),
                       "audience_system_neutral": (P / "audience_system_neutral.txt").read_text(),
                       "audience_system_generic": (P / "audience_system_generic.txt").read_text(),
                       "audience_user": (P / "audience_user.txt").read_text()}}

    model = args.model
    if model is None:
        cand = exp["models"]["audience_primary"]["checkpoint"]
        model = "qwen3:8b" if cand in (None, "TBD_at_pilot") else cand
    if args.backend == "anthropic" and args.model is None:
        model = exp["models"]["cross_check"]["name"]

    LOGS.mkdir(exist_ok=True)
    client = LLMClient(backend=args.backend, model=model, host=args.host,
                       log_path=str(LOGS / "pilot.jsonl"))

    personas = pick_personas(cfg["personas"]["personas"])
    runs = [("ocean", p) for p in personas] + [("neutral", None), ("generic", None)]
    ncalls = len(runs) * len(BLOCKS) * args.reps
    print(f"backend={args.backend}  model={model}")
    print(f"{len(personas)} OCEAN personas + neutral + generic, x {len(BLOCKS)} blocks "
          f"x {args.reps} reps = {ncalls} calls\n")

    rows = []          # {kind, label, block, obj}
    latencies = []
    valid = total = 0
    t_start = time.perf_counter()
    for blk_name, blk in BLOCKS.items():
        ftext = format_audience_block(blk)
        for kind, p in runs:
            label = p["id"] if kind == "ocean" else kind
            for _ in range(args.reps):
                total += 1
                t0 = time.perf_counter()
                obj, err, _ = run_survey(client, p, ftext, cfg, retries=3, agent_kind=kind)
                latencies.append(time.perf_counter() - t0)
                if obj:
                    valid += 1
                    rows.append({"kind": kind, "label": label, "block": blk_name, "obj": obj})
                else:
                    print(f"  INVALID {label} {blk_name}: {err}")
    wall = time.perf_counter() - t_start

    print(f"\nValid completions: {valid}/{total} ({100*valid/total:.0f}%)")

    # runtime, and what it projects to for the full run
    per = statistics.mean(latencies) if latencies else 0.0
    med = statistics.median(latencies) if latencies else 0.0
    FULL_CALLS = 9792 + 500          # audience responses + generator parameter proposals
    proj_h = FULL_CALLS * per / 3600
    print(f"\nRuntime: {wall:.0f}s total for {total} calls  |  "
          f"{per:.1f}s/call mean, {med:.1f}s median")
    print(f"Projected full run ({FULL_CALLS:,} calls at this rate): "
          f"~{proj_h:.1f} h ({proj_h/24:.1f} days) of wall-clock")

    def m(label, blk, q):
        vals = [r["obj"][q] for r in rows if r["label"] == label and r["block"] == blk]
        return statistics.mean(vals) if vals else float("nan")

    def kmean(kind, blk, q):
        vals = [r["obj"][q] for r in rows if r["kind"] == kind and r["block"] == blk]
        return statistics.mean(vals) if vals else float("nan")

    print("\nMean Q2 (arousal) by OCEAN persona x block   [E=extraversion, N=neuroticism]")
    print("  persona          " + "".join(f"{b:>17}" for b in BLOCKS))
    for p in personas:
        tag = f"{p['extraversion'][0].upper()}E {p['neuroticism'][0].upper()}N"
        print(f"  {p['id']:<6} {tag:<8}" + "".join(f"{m(p['id'], b, 'Q2'):>17.1f}" for b in BLOCKS))

    print("\nOCEAN spread vs baselines (arousal Q2, per block):")
    for b in BLOCKS:
        ov = [m(p["id"], b, "Q2") for p in personas]
        ov = [v for v in ov if v == v]
        sd = statistics.pstdev(ov) if len(ov) > 1 else 0.0
        neu, gen = kmean("neutral", b, "Q2"), kmean("generic", b, "Q2")
        print(f"  {b:<16} OCEAN mean={statistics.mean(ov):.1f} SD={sd:.2f} "
              f"range=[{min(ov):.1f},{max(ov):.1f}]   neutral={neu:.1f}  generic={gen:.1f}  "
              f"(base gap={abs(neu-gen):.1f})")

    # Pooled: does persona identity systematically shift arousal, net of the stimulus?
    dev = {p["id"]: [] for p in personas}
    for b in BLOCKS:
        cells = [m(p["id"], b, "Q2") for p in personas]
        cells = [v for v in cells if v == v]
        if not cells:
            continue
        bm = statistics.mean(cells)
        for p in personas:
            v = m(p["id"], b, "Q2")
            if v == v:
                dev[p["id"]].append(v - bm)
    eff = {pid: statistics.mean(d) for pid, d in dev.items() if d}
    pooled = statistics.pstdev(list(eff.values())) if len(eff) > 1 else 0.0
    print(f"\nSystematic persona effect (SD of per-persona arousal deviation, pooled over stimuli): "
          f"{pooled:.2f}")
    print("  near 0 = personas do not systematically differ; > ~0.3 = a real, consistent persona effect.")

    print("\nFeature tracking (OCEAN mean):")
    print(f"  arousal Q2:  bright={kmean('ocean', 'moderate_bright', 'Q2'):.1f}  "
          f"dark={kmean('ocean', 'moderate_dark', 'Q2'):.1f}")
    print(f"  valence Q1:  bright={kmean('ocean', 'moderate_bright', 'Q1'):.1f}  "
          f"dark={kmean('ocean', 'moderate_dark', 'Q1'):.1f}")

    print("\nPASS if: validity ~100%; OCEAN persona SD clearly > 0 off the ceiling;")
    print("bright arousal > dark; OCEAN spread comparable-to-or-larger than the baseline gap.")


if __name__ == "__main__":
    main()