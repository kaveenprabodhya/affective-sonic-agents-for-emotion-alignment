"""Stage 5: the audience harness (H2 + H3 data).

Every stimulus is presented to every agent as a feature block (never audio, never the
target or condition). Agents: the 32 OCEAN personas plus the neutral and generic
baselines. Each agent x stimulus is repeated `reps` times. Output is one tidy row per
response, appended incrementally so the run is resumable after any interruption.

  34 agents x 96 stimuli x 3 reps = 9792 responses.

    python src/audience/run_audience.py --backend ollama
    python src/audience/run_audience.py --backend mock --limit-stimuli 2 --limit-agents 4   # dry run
    python src/audience/run_audience.py --backend ollama --resume                            # continue
"""
import sys
import os
import csv
import json
import time
import random
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT, LOGS                  # noqa: E402
from llm.client import LLMClient                             # noqa: E402
from audience.survey import run_survey                       # noqa: E402
from features.extracts import load_audio, extract_audience_block, format_audience_block  # noqa: E402

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
Q_COLS = [f"Q{i}" for i in range(1, 13)]
FIELDS = (["agent_kind", "persona_id"] + TRAITS +
          ["stimulus_file", "brief", "condition", "target_v", "target_a", "rep"] +
          Q_COLS + ["perceived_v", "perceived_a"])


def build_agents(personas_cfg):
    """32 OCEAN personas + two baselines."""
    agents = []
    for p in personas_cfg["personas"]:
        agents.append({"kind": "ocean", "id": p["id"], "persona": p,
                       "traits": {t: p[t] for t in TRAITS}})
    for base in ("neutral", "generic"):
        agents.append({"kind": base, "id": base, "persona": None,
                       "traits": {t: "" for t in TRAITS}})
    return agents


def build_stimuli(manifest):
    """Each pair yields two stimuli (non-optimised, optimised)."""
    stim = []
    for r in manifest:
        for cond, key in (("non_optimised", "non_optimised"), ("optimised", "optimised")):
            stim.append({"file": r[cond]["file"], "brief": r["brief"], "condition": key,
                         "target": r["target"]})
    return stim


def cell_key(agent_id, stim_file, rep):
    return f"{agent_id}|{stim_file}|{rep}"


def load_done(csv_path):
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(cell_key(row["persona_id"], row["stimulus_file"], row["rep"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip cells already in the output CSV")
    ap.add_argument("--limit-stimuli", type=int)
    ap.add_argument("--limit-agents", type=int)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp = load("experiment.yaml")
    sr = exp["synthesis"]["sample_rate_hz"]
    reps = args.reps or exp["audience"]["repetitions"]
    manifest = json.loads((ROOT / "data" / "stimuli" / "manifest.json").read_text())

    P = ROOT / "config/prompts"
    cfg = {"questionnaire": load("questionnaire.yaml"),
           "personas": load("personas.yaml"),
           "prompts": {"audience_system": (P / "audience_system.txt").read_text(),
                       "audience_system_neutral": (P / "audience_system_neutral.txt").read_text(),
                       "audience_system_generic": (P / "audience_system_generic.txt").read_text(),
                       "audience_user": (P / "audience_user.txt").read_text()}}

    agents = build_agents(cfg["personas"])
    stimuli = build_stimuli(manifest)
    if args.limit_agents:
        agents = agents[:args.limit_agents]
    if args.limit_stimuli:
        stimuli = stimuli[:args.limit_stimuli]

    model = args.model or exp["models"]["audience_primary"]["checkpoint"]
    if args.backend == "ollama" and model in (None, "TBD_at_pilot"):
        model = "qwen3:8b"
    LOGS.mkdir(exist_ok=True)
    client = LLMClient(backend=args.backend, model=model, host=args.host,
                       log_path=str(LOGS / "audience.jsonl"))

    out_dir = ROOT / "data" / "audience"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) if args.out else out_dir / "responses.csv"

    # pre-extract each stimulus's feature block once (not per agent x rep)
    stim_dir = ROOT / "data" / "stimuli"
    print(f"extracting feature blocks for {len(stimuli)} stimuli...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in stimuli:
            y, _sr = load_audio(str(stim_dir / s["file"]), sr=sr)
            s["ftext"] = format_audience_block(extract_audience_block(y, _sr))

    done = load_done(csv_path) if args.resume else set()
    new_file = not csv_path.exists() or (not args.resume and csv_path.stat().st_size == 0)
    if new_file and not args.resume:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # build and shuffle the work list (stateless calls, but randomise order per the design)
    work = [(a, s, rep) for rep in range(reps) for a in agents for s in stimuli]
    random.Random(12345).shuffle(work)
    todo = [w for w in work if cell_key(w[0]["id"], w[1]["file"], w[2]) not in done]

    total = len(work)
    print(f"{len(agents)} agents x {len(stimuli)} stimuli x {reps} reps = {total} responses")
    print(f"already done: {len(done)} | to do now: {len(todo)}\n")

    t0 = time.time()
    written = 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        for i, (a, s, rep) in enumerate(todo, 1):
            obj, err, _ = run_survey(client, a["persona"], s["ftext"], cfg,
                                     retries=exp["audience"]["retries"], agent_kind=a["kind"])
            if obj is None:
                print(f"  INVALID {a['id']} {s['file']} rep{rep}: {err}")
                continue
            row = {"agent_kind": a["kind"], "persona_id": a["id"],
                   **a["traits"],
                   "stimulus_file": s["file"], "brief": s["brief"], "condition": s["condition"],
                   "target_v": s["target"][0], "target_a": s["target"][1], "rep": rep,
                   **{q: obj[q] for q in Q_COLS},
                   "perceived_v": round((obj["Q1"] - 5) / 4, 3),
                   "perceived_a": round((obj["Q2"] - 5) / 4, 3)}
            writer.writerow(row)
            written += 1
            if i % 200 == 0:
                f.flush()
                rate = (time.time() - t0) / i
                eta_h = rate * (len(todo) - i) / 3600
                print(f"  {i}/{len(todo)}  ({rate:.1f}s/resp, ETA {eta_h:.1f}h)")

    print(f"\nwrote {written} responses -> {csv_path}")
    print(f"total in file: {len(load_done(csv_path))}/{total}")
    if len(load_done(csv_path)) < total:
        print("Incomplete - rerun with --resume to continue.")
    else:
        print("Complete. This is the H2/H3 dataset.")


if __name__ == "__main__":
    main()