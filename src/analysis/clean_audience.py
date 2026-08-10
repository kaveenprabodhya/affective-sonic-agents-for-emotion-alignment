#!/usr/bin/env python3
"""Separate the audience responses that match the current briefs from stale ones.

data/audience/responses.csv is appended to, not replaced. After the brief targets
change, the file holds rows scored against the old targets alongside rows scored
against the new ones - and because generation reuses filenames, the old rows
describe audio that no longer exists.

This reports the split and, with --write, saves the current-target rows to a new
file, leaving the original untouched.

    python src/analysis/clean_audience.py
    python src/analysis/clean_audience.py --write
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT      # noqa: E402

TOL = 1e-6
RESPONSES = ROOT / "data" / "audience" / "responses.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the current-target rows to responses_current.csv")
    ap.add_argument("--replace", action="store_true",
                    help="also move the original to responses_all.csv and put the "
                         "cleaned rows at responses.csv")
    args = ap.parse_args()

    if not RESPONSES.exists():
        sys.exit(f"No {RESPONSES}")

    briefs = {b["id"]: (float(b["target"]["valence"]), float(b["target"]["arousal"]))
              for b in load("briefs.yaml")["briefs"]}
    print(f"current briefs.yaml: {len(briefs)} briefs")

    with RESPONSES.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    print(f"responses.csv: {len(rows)} rows\n")

    def matches(r):
        t = briefs.get(r["brief"])
        if t is None:
            return False
        return (abs(float(r["target_v"]) - t[0]) < TOL
                and abs(float(r["target_a"]) - t[1]) < TOL)

    current = [r for r in rows if matches(r)]
    stale = [r for r in rows if not matches(r)]

    print(f"rows matching the current targets : {len(current)}")
    print(f"rows from earlier target sets     : {len(stale)}")

    # which target sets are present, so the history is visible
    sets = Counter()
    for r in rows:
        sets[(r["brief"], round(float(r["target_v"]), 4), round(float(r["target_a"]), 4))] += 1
    per_brief = defaultdict(set)
    for (b, v, a) in sets:
        per_brief[b].add((v, a))
    n_sets = max((len(s) for s in per_brief.values()), default=0)
    print(f"distinct target sets per brief    : {n_sets}")
    if n_sets > 1:
        b = sorted(per_brief)[0]
        print(f"  e.g. {b}: " + "  ".join(f"({v:+.3f},{a:+.3f})" for v, a in sorted(per_brief[b])))

    # is the current-target subset a complete, non-duplicated run?
    print()
    key = lambda r: (r.get("agent_kind"), r.get("persona_id", ""), r["stimulus_file"], r["rep"])
    counts = Counter(key(r) for r in current)
    dupes = {k: c for k, c in counts.items() if c > 1}
    agents = len({(r.get("agent_kind"), r.get("persona_id", "")) for r in current})
    stimuli = len({r["stimulus_file"] for r in current})
    reps = len({r["rep"] for r in current})
    expected = agents * stimuli * reps

    print(f"current subset: {agents} agents x {stimuli} stimuli x {reps} reps = {expected} expected")
    print(f"                {len(current)} rows present, {len(dupes)} duplicated keys")
    if len(current) == expected and not dupes:
        print("                complete and non-duplicated: OK")
    elif dupes:
        print("                DUPLICATES: more than one run shares the current targets.")
        print("                Re-running the audience is the safe option.")
    else:
        print(f"                INCOMPLETE: {expected - len(current)} rows missing.")
        print("                Run: python src/audience/run_audience.py --backend ollama --resume")

    if not (args.write or args.replace):
        print("\n(nothing written; pass --write to save the cleaned rows)")
        return

    out = RESPONSES.parent / "responses_current.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(current)
    print(f"\nwrote {len(current)} rows -> {out}")

    if args.replace:
        allf = RESPONSES.parent / "responses_all.csv"
        RESPONSES.rename(allf)
        out.rename(RESPONSES)
        print(f"original preserved as {allf}")
        print(f"cleaned rows now at   {RESPONSES}")


if __name__ == "__main__":
    main()