"""Rewrite briefs.yaml with targets placed inside the reachable VA region.

Reads models/reachable_va.json (from probe_reachable.py) and the ORIGINAL full-range
briefs (for the per-quadrant spread pattern and descriptions), rescales each target
into the reachable region per axis - respecting the axis asymmetry - and writes a new
briefs.yaml. The original full-range briefs are preserved in briefs_full_range.yaml,
and are always used as the source, so re-running never double-rescales.

    python src/generator/generate_briefs.py
"""
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load, ROOT, CONFIG                # noqa: E402

ORIG_MAX = 0.8   # nominal max target magnitude in the original full-range briefs


def rescale(x, pos_ext, neg_ext):
    ext = pos_ext if x >= 0 else abs(neg_ext)
    return round(x / ORIG_MAX * ext, 3)


def main():
    rp = ROOT / "models" / "reachable_va.json"
    if not rp.exists():
        sys.exit("Run probe_reachable.py first (models/reachable_va.json is missing).")
    r = json.loads(rp.read_text())["report"]
    coach = r.get("coach", "estimator_A")     # recorded by probe_reachable.py
    v_pos, v_neg = r["valence"]["p95"], r["valence"]["p5"]
    a_pos, a_neg = r["arousal"]["p95"], r["arousal"]["p5"]
    print(f"reachable region ({coach}):  valence [{v_neg:+.2f}, {v_pos:+.2f}]   "
          f"arousal [{a_neg:+.2f}, {a_pos:+.2f}]")

    for cond, msg in [
        (v_pos <= 0.03, "valence barely reaches positive - HV quadrants will be weak"),
        (v_neg >= -0.03, "valence barely reaches negative - LV quadrants will be weak"),
        (a_pos <= 0.03, "arousal barely reaches positive - HA quadrants will be weak"),
        (a_neg >= -0.03, "arousal barely reaches negative - LA quadrants will be weak"),
    ]:
        if cond:
            print(f"  WARNING: {msg} (region does not span that corner)")

    # always rescale from the ORIGINAL full-range briefs (idempotent)
    backup = CONFIG / "briefs_full_range.yaml"
    if not backup.exists():
        shutil.copyfile(CONFIG / "briefs.yaml", backup)
    briefs = load("briefs_full_range.yaml")["briefs"]

    lines = ["# 16 brand briefs, 4 per quadrant. Targets rescaled into the reachable VA",
             f"# region measured by probe_reachable.py (synthetic logos scored by {coach}):",
             f"#   reachable valence [{v_neg:+.2f}, {v_pos:+.2f}], arousal [{a_neg:+.2f}, {a_pos:+.2f}].",
             "# Original full-circumplex targets are preserved in briefs_full_range.yaml.",
             "scale: [-1, 1]", "n_briefs: 16", "", "briefs:"]
    print(f"\n{'brief':6} {'quadrant':8} {'original':>16} {'rescaled':>16}")
    for b in briefs:
        ov, oa = b["target"]["valence"], b["target"]["arousal"]
        nv, na = rescale(ov, v_pos, v_neg), rescale(oa, a_pos, a_neg)
        lines += [f"  - id: {b['id']}",
                  f"    quadrant: {b['quadrant']}",
                  f"    target: {{valence: {nv}, arousal: {na}}}",
                  f'    brand_description: "{b["brand_description"]}"']
        print(f"{b['id']:6} {b['quadrant']:8} ({ov:+.2f},{oa:+.2f})   ({nv:+.2f},{na:+.2f})")

    (CONFIG / "briefs.yaml").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(briefs)} rescaled briefs to config/briefs.yaml "
          f"(originals in briefs_full_range.yaml)")


if __name__ == "__main__":
    main()