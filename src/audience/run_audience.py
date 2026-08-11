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
import hashlib
import time
import random
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT, LOGS                  # noqa: E402
from llm.client import LLMClient                             # noqa: E402
from llm.client import call_seed
from audience.survey import run_survey                       # noqa: E402
from features.extracts import load_audio, extract_audience_block, format_audience_block  # noqa: E402

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
Q_COLS = [f"Q{i}" for i in range(1, 13)]
FIELDS = (["agent_kind", "persona_id"] + TRAITS +
          ["stimulus_file", "brief", "condition", "target_v", "target_a",
           "stimulus_sha256", "rep"] +
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


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_done(csv_path):
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(cell_key(row["persona_id"], row["stimulus_file"], row["rep"]))
    return done


def stale_rows(csv_path, stimuli, tol=1e-6):
    """Rows in an existing CSV that do not describe the current stimuli.

    Generation reuses filenames, so old rows carry the same stimulus_file while
    describing audio that no longer exists. Resuming onto them merges two studies
    into one file with nothing left to separate them afterwards.

    The check prefers the recorded sha256, because the hash identifies the audio
    a response actually heard. The target only identifies the brief, so it misses
    a regeneration that left the briefs alone - and since LLM output is not
    deterministic, every regeneration produces different audio under unchanged
    filenames and unchanged targets. Rows written before the hash column existed
    fall back to the target comparison, which is weaker but better than nothing.
    """
    if not csv_path.exists():
        return 0, 0
    want_t = {s["file"]: (float(s["target"][0]), float(s["target"][1])) for s in stimuli}
    want_h = {s["file"]: s.get("sha256") for s in stimuli}
    stale = total = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            total += 1
            fn = row["stimulus_file"]
            if fn not in want_t:
                stale += 1
                continue
            have_h = (row.get("stimulus_sha256") or "").strip()
            if have_h and want_h.get(fn):
                if have_h != want_h[fn]:
                    stale += 1
                continue
            t = want_t[fn]
            if (abs(float(row["target_v"]) - t[0]) > tol
                    or abs(float(row["target_a"]) - t[1]) > tol):
                stale += 1
    return stale, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip cells already in the output CSV")
    ap.add_argument("--force", action="store_true",
                    help="resume even when the existing rows were scored against "
                         "different stimuli (not recommended)")
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
            path = stim_dir / s["file"]
            s["sha256"] = sha256_of(path)
            y, _sr = load_audio(str(path), sr=sr)
            s["ftext"] = format_audience_block(extract_audience_block(y, _sr))

    # A run without --resume starts a new file. The previous one is kept under a
    # timestamped name rather than deleted:
    if args.resume:
        stale, total = stale_rows(csv_path, stimuli)
        if stale and not args.force:
            sys.exit(
                f"\nRefusing to resume: {stale} of {total} rows in {csv_path.name} were\n"
                f"scored against different stimuli than the current manifest (the brief\n"
                f"targets have changed since they were written).\n\n"
                f"Filenames are reused across generations, so resuming would mix two\n"
                f"studies into one file with no way to separate them afterwards.\n\n"
                f"  start fresh:      python {Path(__file__).name} --backend <backend>\n"
                f"  inspect/salvage:  python src/analysis/clean_audience.py\n"
                f"  override anyway:  --force\n")
        if stale and args.force:
            print(f"WARNING: --force with {stale}/{total} rows from different stimuli.\n")
        done = load_done(csv_path)
    else:
        done = set()
        if csv_path.exists() and csv_path.stat().st_size > 0:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = csv_path.with_name(f"{csv_path.stem}_{stamp}.csv")
            csv_path.rename(backup)
            print(f"existing responses moved to {backup.name} "
                  f"(pass --resume to continue one instead)")
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
                                     retries=exp["audience"]["retries"], agent_kind=a["kind"],
                                     seed=call_seed(a["id"], s["file"], rep))
            if obj is None:
                print(f"  INVALID {a['id']} {s['file']} rep{rep}: {err}")
                continue
            row = {"agent_kind": a["kind"], "persona_id": a["id"],
                   **a["traits"],
                   "stimulus_file": s["file"], "brief": s["brief"], "condition": s["condition"],
                   "target_v": s["target"][0], "target_a": s["target"][1],
                   "stimulus_sha256": s.get("sha256", ""), "rep": rep,
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