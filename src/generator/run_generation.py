"""Run the generation stage: optimise every brief x run, emit the matched
non-optimised / optimised stimulus pair, and write a manifest.

    python src/generator/run_generation.py --soundfont assets/soundfronts/GeneralUser-GS.sf2
    python src/generator/run_generation.py --backend mock --limit 2   # quick dry run

Estimator A drives the loop (coach). Estimator B is never touched here (it judges
H1 later, on these frozen stimuli).
"""
import sys
import os
import json
import glob
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT, LOGS                 # noqa: E402
from llm.client import LLMClient                            # noqa: E402
from estimators.model import load as load_estimator         # noqa: E402
from generator.loop import OptimisationController           # noqa: E402


def resolve_soundfont(arg, exp):
    if arg:
        return arg
    cfg_path = exp.get("synthesis", {}).get("soundfont", {}).get("path")
    if cfg_path:
        p = ROOT / cfg_path
        if p.exists():
            return str(p)
    hits = glob.glob(str(ROOT / "assets" / "**" / "*.sf2"), recursive=True)
    if hits:
        return hits[0]
    sys.exit("No soundfont found. Pass --soundfont, set synthesis.soundfont.path in "
             "experiment.yaml, or drop a .sf2 under assets/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundfont", help="Path to the .sf2 (else config path or assets/**/*.sf2)")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, help="Only the first N briefs (quick run)")
    ap.add_argument("--briefs", help="Comma-separated brief IDs, e.g. B01,B05,B09,B13 "
                                     "(pilot across quadrants; overrides --limit)")
    args = ap.parse_args()

    exp = load("experiment.yaml")
    briefs = load("briefs.yaml")["briefs"]
    if args.briefs:
        want = [b.strip() for b in args.briefs.split(",")]
        by_id = {b["id"]: b for b in briefs}
        missing = [w for w in want if w not in by_id]
        if missing:
            sys.exit(f"Unknown brief id(s): {', '.join(missing)}")
        briefs = [by_id[w] for w in want]
    elif args.limit:
        briefs = briefs[:args.limit]

    sf = resolve_soundfont(args.soundfont, exp)
    duration = exp.get("synthesis", {}).get("duration_s", 3.0)
    sr = exp.get("synthesis", {}).get("sample_rate_hz", 22050)
    opt = exp["generation"]["optimisation"]
    runs_per = exp["generation"]["runs_per_brief"]
    draw_retries = exp["generation"]["retries"]

    model = args.model or exp["models"]["generator_primary"]["checkpoint"]
    if args.backend == "ollama" and model in (None, "TBD_at_pilot"):
        model = "qwen3:8b"
    LOGS.mkdir(exist_ok=True)
    client = LLMClient(backend=args.backend, model=model, host=args.host,
                       log_path=str(LOGS / "generation.jsonl"))
    predict_a, meta_a = load_estimator("estimator_A")
    templates = {"initial": (ROOT / "config/prompts/generator_initial.txt").read_text(),
                 "revision": (ROOT / "config/prompts/generator_revision.txt").read_text()}

    out_dir = ROOT / "data" / "stimuli"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl = OptimisationController(client, predict_a, sf, templates,
                                  threshold=opt["threshold"], iteration_cap=opt["iteration_cap"],
                                  duration=duration, sr=sr, retries=draw_retries,
                                  require_quadrant=opt.get("require_quadrant", True))

    print(f"soundfont: {sf}")
    print(f"{len(briefs)} briefs x {runs_per} runs, threshold={opt['threshold']}, "
          f"quadrant_required={opt.get('require_quadrant', True)}, "
          f"cap={opt['iteration_cap']}, model={model}\n")

    manifest = []
    for brief in briefs:
        for run_idx in range(runs_per):
            res = None
            for _ in range(draw_retries + 1):       # re-draw a run that fails to produce a pair
                res = ctrl.run(brief, run_idx, str(out_dir))
                if res:
                    break
            if not res:
                print(f"  FAILED {brief['id']} run{run_idx} (no valid pair after retries)")
                continue
            manifest.append(res)
            flag = "*" if res["reached_threshold"] else " "
            qflag = "Q" if res["quadrant_held"] else "x"
            print(f" {flag}{qflag} {brief['id']} run{run_idx}: "
                  f"non-opt d={res['non_optimised']['distance']:.3f} -> "
                  f"opt d={res['optimised']['distance']:.3f}  ({res['iterations']} iters)")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    reached = sum(r["reached_threshold"] for r in manifest)
    within  = sum(r["reached_distance"] for r in manifest)
    quad    = sum(r["quadrant_held"] for r in manifest)
    print(f"\n{len(manifest)} pairs, {2*len(manifest)} stimuli -> {out_dir}")
    print(f"{reached}/{len(manifest)} runs met BOTH criteria; "
          f"{within}/{len(manifest)} met the distance threshold; "
          f"{quad}/{len(manifest)} held the intended quadrant.")
    print("manifest.json written. (* = both criteria met, Q = quadrant held)")


if __name__ == "__main__":
    main()