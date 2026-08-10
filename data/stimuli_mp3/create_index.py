#!/usr/bin/env python3
"""Build the stimuli listening page.

Reads the generation manifest, the brand briefs and (if present) the held-out
Estimator B scores, then writes a single self-contained index.html next to the
MP3s. Each brief/run pair becomes one card: the brand description that produced
it, both audio versions, what the two estimators made of them, and — the part
that explains how they sound — exactly which synth parameters changed.

Run from the project root:
    python data/stimuli_mp3/create_index.py                        # incumbent judge
    python data/stimuli_mp3/create_index.py --judge estimator_B2   # selected judge
"""

import argparse
import csv
import re
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MP3_DIR = ROOT / "data" / "stimuli_mp3"
MANIFEST = ROOT / "data" / "stimuli" / "manifest.json"
BRIEFS = ROOT / "config" / "briefs.yaml"
ANALYSIS_DIR = ROOT / "data" / "analysis"

# Quadrant hues follow the circumplex: warm where arousal is high, cool where it
# is low; saturated where valence is negative. Colour encodes position, not decoration.
QUAD = {
    "HV_HA": ("#C2621B", "High valence · High arousal", "excited, energetic"),
    "LV_HA": ("#A62F47", "Low valence · High arousal", "tense, agitated"),
    "LV_LA": ("#2E5E8E", "Low valence · Low arousal", "subdued, melancholy"),
    "HV_LA": ("#3D7A5A", "High valence · Low arousal", "calm, contented"),
}

# Parameters shown in this order; anything not listed still renders, after these.
PARAM_ORDER = ["tempo_bpm", "notes_per_beat", "instrument", "mode", "contour",
               "pitch_center_midi", "pitch_range", "dynamics", "articulation"]

PARAM_LABEL = {
    "tempo_bpm": "Tempo",
    "notes_per_beat": "Notes per beat",
    "instrument": "Instrument",
    "mode": "Mode",
    "contour": "Contour",
    "pitch_center_midi": "Pitch centre",
    "pitch_range": "Pitch range",
    "dynamics": "Dynamics",
    "articulation": "Articulation",
}

PARAM_UNIT = {"tempo_bpm": " BPM", "pitch_center_midi": " MIDI", "pitch_range": " st"}

# Plot domain. Computed from the data at build time so no point is ever clamped
# to the frame edge: a clamped dot silently misreports its position. All plots on
# the page share one scale, so a compressed judge visibly stays compressed next to
# a judge that spreads.
PLOT_PX = 168             # plots keep this size whatever the estimator count; the strip scrolls
DOMAIN_MIN = 0.2          # never zoom in tighter than this
DOMAIN_HEADROOM = 1.1     # widen past the outermost point so nothing sits on the frame


def load_briefs(path):
    """Minimal YAML reader for the flat brief structure. Avoids a PyYAML dependency
    so the page can be rebuilt even from a bare checkout."""
    try:
        import yaml
        return {b["id"]: b for b in yaml.safe_load(path.read_text())["briefs"]}
    except ImportError:
        pass

    briefs, cur = {}, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("- id:"):
            if cur:
                briefs[cur["id"]] = cur
            cur = {"id": line.split(":", 1)[1].strip().strip('"\''), "target": {}}
        elif cur is None:
            continue
        elif line.startswith("quadrant:"):
            cur["quadrant"] = line.split(":", 1)[1].strip().strip('"\'')
        elif line.startswith("brand_description:"):
            cur["brand_description"] = line.split(":", 1)[1].strip().strip('"\'')
        elif line.startswith("valence:"):
            cur["target"]["valence"] = float(line.split(":", 1)[1])
        elif line.startswith("arousal:"):
            cur["target"]["arousal"] = float(line.split(":", 1)[1])
    if cur:
        briefs[cur["id"]] = cur
    return briefs


def split_name(name):
    """estimator_B2 -> ("B", 2);  estimator_B -> ("B", 1)."""
    m = re.match(r"^estimator_([A-Za-z]+)(\d*)$", name)
    if not m:
        return name, 1
    return m.group(1), int(m.group(2) or 1)


def display_name(name):
    """What the page calls an estimator.

    The trailing version number is an artefact of how the model was produced, not
    something a reader needs: estimator_A2 is simply the coach this study used, so
    the page calls it Estimator A. The real filename appears once, in the roles
    panel, so anything shown here can still be traced back to models/.
    """
    base, _ = split_name(name)
    return f"Estimator {base}"


def keep_final(found):
    """Among estimators sharing a base letter, keep only the highest version.

    A later version exists precisely because it replaced the earlier one, so
    showing both would put a superseded estimator beside the one that replaced it
    under almost the same label.
    """
    best = {}
    for name, path in found:
        base, ver = split_name(name)
        if base not in best or ver > split_name(best[base][0])[1]:
            best[base] = (name, path)
    order = [n for n, _ in found]
    return sorted(best.values(), key=lambda t: order.index(t[0]))


def discover_judges(only=None, coach=None, show_all=False):
    """Every held-out judge that has been scored, in a stable order.

    h1_estimator_b.csv            -> estimator_B   (the incumbent)
    h1_estimator_b_<name>.csv     -> <name>

    The coach is excluded. Scoring the stimuli with the coach is a useful
    diagnostic elsewhere, but it already has its own column here, taken from the
    estimates the loop recorded in the manifest. Listing it again would show the
    same model twice and label the second copy "held out", which it was not.
    Naming it in --judges overrides this.
    """
    found = []
    inc = ANALYSIS_DIR / "h1_estimator_b.csv"
    if inc.exists():
        found.append(("estimator_B", inc))
    for f in sorted(ANALYSIS_DIR.glob("h1_estimator_b_*.csv")):
        found.append((f.stem.replace("h1_estimator_b_", ""), f))
    if only:
        found = [(n, f) for n, f in found if n in only]
    else:
        if not show_all:
            found = keep_final(found)
        if coach:
            found = [(n, f) for n, f in found if n != coach]
    return [(n, display_name(n), load_est_b(f)) for n, f in found]


def load_est_b(path):
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open()):
        out[(row["brief"], int(row["run"]))] = {
            "nonopt": (float(row["nonopt_B_v"]), float(row["nonopt_B_a"])),
            "opt": (float(row["opt_B_v"]), float(row["opt_B_a"])),
            "nonopt_dist": float(row["nonopt_B_dist"]),
            "opt_dist": float(row["opt_B_dist"]),
        }
    return out


def same_sign(est, target):
    for e, t in zip(est, target):
        if t == 0:
            continue
        if e * t <= 0:
            return False
    return True


def to_xy(v, a, dom, size=168, pad=14):
    """Valence on x, arousal on y (inverted for SVG's downward y)."""
    span = size - 2 * pad
    x = pad + (v + dom) / (2 * dom) * span
    y = pad + (dom - a) / (2 * dom) * span
    return round(x, 1), round(y, 1)


def plot(target, nonopt, opt, colour, dom, size=168, pad=14):
    """The signature element: where the stimulus sat, where it moved, where it
    was aimed. Rendered identically on every card so cards are comparable."""
    tx, ty = to_xy(*target, dom=dom, size=size, pad=pad)
    nx, ny = to_xy(*nonopt, dom=dom, size=size, pad=pad)
    ox, oy = to_xy(*opt, dom=dom, size=size, pad=pad)
    mid = size / 2
    moved = abs(nx - ox) > 1.5 or abs(ny - oy) > 1.5

    arrow = ""
    if moved:
        arrow = (f'<line class="mv" x1="{nx}" y1="{ny}" x2="{ox}" y2="{oy}" '
                 f'marker-end="url(#ah{colour.lstrip("#")})"/>')

    return f'''<svg class="plot" viewBox="0 0 {size} {size}" role="img"
     aria-label="Valence-arousal plot: target, first candidate, best candidate">
  <defs><marker id="ah{colour.lstrip('#')}" markerWidth="7" markerHeight="7"
     refX="5.5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="{colour}"/></marker></defs>
  <rect x="{pad}" y="{pad}" width="{size-2*pad}" height="{size-2*pad}" class="frame"/>
  <line x1="{mid}" y1="{pad}" x2="{mid}" y2="{size-pad}" class="axis"/>
  <line x1="{pad}" y1="{mid}" x2="{size-pad}" y2="{mid}" class="axis"/>
  <circle cx="{tx}" cy="{ty}" r="7" class="target" stroke="{colour}"/>
  <circle cx="{tx}" cy="{ty}" r="1.6" fill="{colour}"/>
  {arrow}
  <circle cx="{nx}" cy="{ny}" r="4.5" class="dot-non"/>
  <circle cx="{ox}" cy="{oy}" r="4.5" class="dot-opt" fill="{colour}"/>
</svg>'''


def xy(p):
    return f"{p[0]:+.2f},{p[1]:+.2f}"


def pcol(title, sub, target, non, opt, colour, size, dom):
    """One plot column: heading, plot, and the coordinates it is drawing.

    The numbers are shown so the picture can be checked against the CSV rather
    than trusted.
    """
    return (f'<div class="pcol">'
            f'<h4>{title} <small>{sub} &middot; &plusmn;{dom:.2f}</small></h4>'
            f'{plot(target, non, opt, colour, dom, size=size)}'
            f'<div class="coords">{xy(non)} <i>&rarr;</i> {xy(opt)}</div></div>')


def fmt(key, val):
    return f"{val}{PARAM_UNIT.get(key, '')}"


def param_rows(p_non, p_opt):
    keys = [k for k in PARAM_ORDER if k in p_non or k in p_opt]
    keys += [k for k in sorted(set(p_non) | set(p_opt)) if k not in keys]
    changed, unchanged = [], []
    for k in keys:
        a, b = p_non.get(k), p_opt.get(k)
        label = PARAM_LABEL.get(k, k.replace("_", " ").capitalize())
        if a != b:
            direction = ""
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                direction = " up" if b > a else " down"
            changed.append(
                f'<tr class="chg{direction}"><th>{label}</th>'
                f'<td class="was">{fmt(k, a)}</td>'
                f'<td class="arrow" aria-hidden="true">&rarr;</td>'
                f'<td class="now">{fmt(k, b)}</td></tr>')
        else:
            unchanged.append(f'<tr><th>{label}</th><td colspan="3">{fmt(k, a)}</td></tr>')
    return changed, unchanged


def card(run, brief, judges, idx, doms, coach_label):
    bid, ridx = run["brief"], run["run"]
    colour, quad_name, quad_feel = QUAD.get(brief.get("quadrant", ""), ("#555", "", ""))
    target = tuple(run["target"])
    non, opt = run["non_optimised"], run["optimised"]
    n_est, o_est = tuple(non["est"]), tuple(opt["est"])
    tied = non["file"] == opt["file"] or (n_est == o_est and non["distance"] == opt["distance"])

    changed, unchanged = param_rows(non["params"], opt["params"])
    jrows = [(label, data.get((bid, ridx))) for _n, label, data in judges]

    held = opt.get("quadrant_ok")
    if held is None:
        held = same_sign(o_est, target)
    badges = []
    if tied:
        badges.append('<span class="badge tie">no change — best was the first candidate</span>')
    badges.append(f'<span class="badge {"held" if held else "crossed"}">'
                  f'{"held quadrant" if held else "crossed the axis"} '
                  f'<em>per {coach_label}</em></span>')
    if run.get("reached_threshold"):
        badges.append('<span class="badge met">met both stopping criteria</span>')

    def dist_pair(label, dn, do):
        if dn is None:
            return ""
        delta = dn - do
        arrow = "closer" if delta > 0.0005 else ("further" if delta < -0.0005 else "no change")
        cls = "good" if delta > 0.0005 else ("bad" if delta < -0.0005 else "flat")
        return (f'<div class="dist"><span class="dl">{label}</span>'
                f'<span class="dv">{dn:.3f} <i>&rarr;</i> {do:.3f}</span>'
                f'<span class="dd {cls}">{arrow}</span></div>')

    changed_html = "".join(changed) or (
        '<tr class="none"><td colspan="4">No parameters changed.</td></tr>')

    more = ""
    if unchanged:
        more = (f'<details class="more"><summary>Unchanged parameters '
                f'({len(unchanged)})</summary><table class="params">'
                f'{"".join(unchanged)}</table></details>')

    judge_plots = "".join(
        pcol(label, "held out", target, d["nonopt"], d["opt"], colour, PLOT_PX, doms[label])
        for label, d in jrows if d)
    judge_dists = "".join(
        dist_pair(f'{label} — independent', d["nonopt_dist"], d["opt_dist"])
        for label, d in jrows if d)

    return f'''<article class="card" data-quad="{brief.get('quadrant','')}"
         data-brief="{bid}" data-held="{'1' if held else '0'}"
         data-tied="{'1' if tied else '0'}" style="--accent:{colour}">
  <header class="chead">
    <div class="who">
      <span class="num">{idx:02d}</span>
      <h3>{bid} <span class="run">run {ridx}</span></h3>
      <span class="quad">{quad_name} <em>{quad_feel}</em></span>
      <span class="tgt">target {xy(target)}</span>
    </div>
    <div class="badges">{''.join(badges)}</div>
  </header>

  <p class="brief">{brief.get('brand_description', '—')}</p>

  <div class="body">
    <div class="left">
      <div class="listen">
        <div class="track">
          <div class="tlab"><strong>First candidate</strong>
            <span>non-optimised · iteration 0</span></div>
          <audio controls preload="none" src="{Path(non['file']).stem}.mp3"></audio>
        </div>
        <div class="track">
          <div class="tlab"><strong>Best candidate</strong>
            <span>optimised · iteration {opt.get('iteration', 0)} of {run['iterations']}</span></div>
          <audio controls preload="none" src="{Path(opt['file']).stem}.mp3"></audio>
        </div>
      </div>

      <div class="dists">
        {dist_pair(f'{coach_label} — coach', non['distance'], opt['distance'])}
        {judge_dists}
      </div>
    </div>

    <div class="plots">
      {pcol(coach_label, "coach", target, n_est, o_est, colour, PLOT_PX, doms[coach_label])}
      {judge_plots}
    </div>
  </div>

  <div class="whatchanged">
    <h4>What changed in the synthesiser</h4>
    <table class="params">{changed_html}</table>
    {more}
  </div>
</article>'''


def build(only=None, scale="shared", show_all=False):
    if not MANIFEST.exists():
        sys.exit(f"No manifest at {MANIFEST}. Run the generation stage first.")

    runs = json.loads(MANIFEST.read_text())
    briefs = load_briefs(BRIEFS)

    # The coach that actually produced these estimates. Recorded per run by
    # run_generation.py; older manifests predate that field.
    coach = next((r["coach"] for r in runs if r.get("coach")), "estimator_A")
    coach_label = display_name(coach)

    judges = discover_judges(only, coach, show_all)

    def dom_of(vals):
        m = max((abs(v) for v in vals), default=DOMAIN_MIN)
        return max(DOMAIN_MIN, round(m * DOMAIN_HEADROOM + 0.049, 1))

    tgt = [c for r in runs for c in r["target"]]
    per = {coach_label: tgt + [c for r in runs
                               for c in list(r["non_optimised"]["est"]) + list(r["optimised"]["est"])]}
    for _n, lab, data in judges:
        per[lab] = tgt + [c for d in data.values() for c in list(d["nonopt"]) + list(d["opt"])]

    if scale == "shared":
        one = dom_of([c for v in per.values() for c in v])
        doms = {k: one for k in per}
    else:
        doms = {k: dom_of(v) for k, v in per.items()}

    missing = [r for r in runs if not (MP3_DIR / f"{Path(r['optimised']['file']).stem}.mp3").exists()]
    if missing:
        print(f"warning: {len(missing)} pairs have no MP3 yet — run convert_2_mp3.sh")

    runs.sort(key=lambda r: (r["brief"], r["run"]))
    cards = "\n".join(card(r, briefs.get(r["brief"], {}), judges, i, doms, coach_label)
                      for i, r in enumerate(runs, 1))

    held = sum(1 for r in runs
               if r["optimised"].get("quadrant_ok",
                                     same_sign(tuple(r["optimised"]["est"]), tuple(r["target"]))))
    tied = sum(1 for r in runs
               if r["non_optimised"]["est"] == r["optimised"]["est"]
               and r["non_optimised"]["distance"] == r["optimised"]["distance"])

    def role_card(name, label, kind):
        artefact = f' <span class="artefact">models/{name}</span>'
        if kind == "coach":
            tag, body = "coach", (
                "Guides the optimisation loop. It scores every candidate, reports the gap "
                "to the target back to the language model, and decides when the loop stops. "
                "It sees everything, so agreeing with it is not evidence that a logo carries "
                "the intended emotion.")
        elif name == "estimator_B":
            tag, body = "held out", (
                "The judge specified before the study ran. Never consulted during generation: "
                "it scores the finished audio only, so its verdict is independent of the search.")
        else:
            tag, body = "held out", (
                "An additional held-out judge, chosen by the architecture comparison in "
                "models/selection/. Never consulted during generation.")
        return (f'<div class="role"><h3>{label} <span class="tag {tag.replace(" ", "")}">'
                f'{tag}</span>{artefact}</h3><p>{body}</p></div>')

    role_cards = role_card(coach, coach_label, "coach") + "".join(
        role_card(n, lab, "judge") for n, lab, d in judges if d)

    scale_note = (
        f"Every plot shares one scale, &plusmn;{max(doms.values()):.2f} of the &plusmn;1 "
        f"circumplex, so a judge whose dots barely move really is barely moving."
        if scale == "shared" else
        "Each estimator is scaled to its own range — read the axis label before comparing "
        "columns, because the columns are not on the same scale.")
    quads = sorted({b.get("quadrant") for b in briefs.values() if b.get("quadrant")})
    quad_btns = "".join(
        f'<button class="chip" data-f="quad" data-v="{q}" style="--c:{QUAD.get(q,("#555",))[0]}">'
        f'{q.replace("_", " / ")}</button>' for q in quads)

    html = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonic Logo Stimuli</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --ink: #14181D; --ink-2: #4A5560; --ink-3: #79858F;
  --ground: #FAFBFC; --panel: #FFFFFF; --line: #E1E6EA; --line-2: #EDF1F4;
  --good: #2E7D5B; --bad: #B03A3A;
  --sans: 'Space Grotesk', system-ui, -apple-system, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 1.5;
  -webkit-font-smoothing: antialiased; }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 0 24px 96px; }}

header.top {{ padding: 56px 0 28px; border-bottom: 2px solid var(--ink); margin-bottom: 28px; }}
h1 {{ font-size: clamp(30px, 5vw, 46px); font-weight: 700; letter-spacing: -0.025em;
  margin: 0 0 10px; }}
.sub {{ color: var(--ink-2); max-width: 62ch; margin: 0 0 22px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 28px; font-family: var(--mono);
  font-size: 13px; }}
.stats div {{ display: flex; flex-direction: column; gap: 2px; }}
.stats b {{ font-size: 21px; font-weight: 500; }}
.stats span {{ color: var(--ink-3); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.08em; }}

.legend {{ display: grid; grid-template-columns: auto 1fr; gap: 18px 22px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 26px; align-items: center; }}
.legend h2 {{ grid-column: 1 / -1; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--ink-3); margin: 0; font-weight: 500; }}
.legend p {{ margin: 0; color: var(--ink-2); font-size: 14px; }}
.legend .keys {{ grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 20px;
  font-size: 13px; color: var(--ink-2); padding-top: 4px;
  border-top: 1px solid var(--line-2); }}
.key {{ display: flex; align-items: center; gap: 7px; }}
.swatch {{ width: 13px; height: 13px; border-radius: 50%; flex: none; }}
.sw-t {{ border: 2px solid var(--ink-2); background: none; }}
.sw-n {{ background: none; border: 2px solid var(--ink-3); }}
.sw-o {{ background: var(--ink); }}

.roles {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 18px; }}
.roles h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-3); margin: 0 0 14px; font-weight: 500; }}
.rgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 18px; }}
.role h3 {{ margin: 0 0 5px; font-size: 15px; font-weight: 700; display: flex;
  align-items: baseline; gap: 8px; flex-wrap: wrap; }}
.role p {{ margin: 0; font-size: 13.5px; color: var(--ink-2); }}
.tag {{ font-family: var(--mono); font-size: 10px; font-weight: 400; padding: 2px 7px;
  border-radius: 4px; text-transform: uppercase; letter-spacing: 0.06em; }}
.tag.coach {{ background: #FDF0E4; color: #A05418; border: 1px solid #F0D8BF; }}
.tag.heldout {{ background: #EAF1F8; color: #2E5E8E; border: 1px solid #CEDEEE; }}
.artefact {{ font-family: var(--mono); font-size: 10.5px; font-weight: 400;
  color: var(--ink-3); }}
.convention {{ margin: 16px 0 0; padding-top: 13px; border-top: 1px solid var(--line-2);
  font-size: 13.5px; color: var(--ink-2); }}

.filters {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  margin-bottom: 24px; position: sticky; top: 0; background: var(--ground);
  padding: 12px 0; z-index: 5; border-bottom: 1px solid var(--line); }}
.filters .lab {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--ink-3); margin-right: 4px; }}
.chip {{ font: 500 13px var(--sans); background: var(--panel); color: var(--ink-2);
  border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px;
  cursor: pointer; }}
.chip:hover {{ border-color: var(--ink-3); color: var(--ink); }}
.chip[aria-pressed="true"] {{ background: var(--c, var(--ink)); color: #fff;
  border-color: var(--c, var(--ink)); }}
.chip:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 2px; }}
.count {{ margin-left: auto; font-family: var(--mono); font-size: 12px; color: var(--ink-3); }}

.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 22px 24px 20px; margin-bottom: 18px; border-left: 4px solid var(--accent); }}
.chead {{ display: flex; justify-content: space-between; align-items: flex-start;
  gap: 16px; flex-wrap: wrap; }}
.who {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.num {{ font-family: var(--mono); font-size: 12px; color: var(--ink-3); }}
.chead h3 {{ margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -0.01em; }}
.run {{ font-weight: 400; color: var(--ink-3); font-size: 15px; }}
.quad {{ font-size: 12.5px; color: var(--accent); font-weight: 500; }}
.quad em {{ color: var(--ink-3); font-style: normal; font-weight: 400; }}
.badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.badge {{ font-family: var(--mono); font-size: 10.5px; padding: 3px 9px;
  border-radius: 4px; border: 1px solid var(--line); color: var(--ink-2); }}
.badge.held {{ color: var(--good); border-color: #BFE0CE; background: #F1F9F4; }}
.badge.crossed {{ color: var(--bad); border-color: #EFC9C9; background: #FDF3F3; }}
.badge em {{ font-style: normal; opacity: 0.66; }}
.badge.tie {{ color: var(--ink-3); }}
.badge.met {{ color: var(--ink); border-color: var(--ink-3); }}

.brief {{ margin: 14px 0 18px; font-size: 15.5px; color: var(--ink-2);
  max-width: 74ch; padding-left: 14px; border-left: 2px solid var(--line); }}

.body {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 28px;
  align-items: start; margin-top: 4px; }}
.left {{ min-width: 0; }}
.listen {{ display: flex; flex-direction: column; gap: 12px; }}
.track {{ min-width: 0; }}
.tlab {{ display: flex; align-items: baseline; gap: 9px; margin-bottom: 5px;
  flex-wrap: wrap; }}
.tlab strong {{ font-size: 14px; }}
.tlab span {{ font-family: var(--mono); font-size: 11px; color: var(--ink-3); }}
audio {{ width: 100%; max-width: 420px; height: 36px; }}

.dists {{ margin-top: 16px; border-top: 1px solid var(--line-2); padding-top: 12px;
  display: flex; flex-direction: column; gap: 7px; }}
.dist {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
.dl {{ font-size: 12.5px; color: var(--ink-2); min-width: 24ch; }}
.dv {{ font-family: var(--mono); font-size: 13px; }}
.dv i {{ color: var(--ink-3); font-style: normal; }}
.dd {{ font-family: var(--mono); font-size: 11px; padding: 1px 7px; border-radius: 3px; }}
.dd.good {{ color: var(--good); background: #F1F9F4; }}
.dd.bad {{ color: var(--bad); background: #FDF3F3; }}
.dd.flat {{ color: var(--ink-3); background: var(--line-2); }}

.plots {{ display: flex; gap: 16px; padding-bottom: 6px; max-width: 100%;
  overflow-x: auto; scroll-snap-type: x proximity;
  scrollbar-width: thin; scrollbar-color: var(--line) transparent; }}
.plots::-webkit-scrollbar {{ height: 7px; }}
.plots::-webkit-scrollbar-track {{ background: var(--line-2); border-radius: 4px; }}
.plots::-webkit-scrollbar-thumb {{ background: var(--line); border-radius: 4px; }}
.plots::-webkit-scrollbar-thumb:hover {{ background: var(--ink-3); }}
.pcol {{ flex: 0 0 auto; scroll-snap-align: start; }}
.pcol h4 {{ margin: 0 0 4px; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--ink-3); font-weight: 500; }}
.pcol h4 small {{ text-transform: none; letter-spacing: 0; font-size: 10.5px;
  color: var(--ink-3); opacity: 0.75; }}
.plot {{ width: 168px; height: 168px; display: block; }}
.coords {{ font-family: var(--mono); font-size: 10.5px; color: var(--ink-3);
  text-align: center; margin-top: 3px; }}
.coords i {{ font-style: normal; opacity: 0.6; }}
.tgt {{ font-family: var(--mono); font-size: 11.5px; color: var(--ink-3); }}
.frame {{ fill: #FCFDFD; stroke: var(--line); }}
.axis {{ stroke: var(--line); stroke-dasharray: 3 3; }}
.target {{ fill: none; stroke-width: 1.5; stroke-dasharray: 3 2.5; }}
.dot-non {{ fill: #fff; stroke: var(--ink-3); stroke-width: 2; }}
.dot-opt {{ stroke: #fff; stroke-width: 1.5; }}
.mv {{ stroke: var(--accent); stroke-width: 1.5; opacity: 0.65; }}

.whatchanged {{ margin-top: 20px; border-top: 1px solid var(--line-2); padding-top: 14px; }}
.whatchanged h4 {{ margin: 0 0 9px; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--ink-3); font-weight: 500; }}
table.params {{ border-collapse: collapse; font-size: 13.5px; }}
table.params th {{ text-align: left; font-weight: 400; color: var(--ink-2);
  padding: 3px 22px 3px 0; white-space: nowrap; }}
table.params td {{ font-family: var(--mono); font-size: 13px; padding: 3px 14px 3px 0; }}
td.was {{ color: var(--ink-3); }}
td.arrow {{ color: var(--ink-3); padding: 0 4px 0 0; }}
td.now {{ font-weight: 500; }}
tr.up td.now::after {{ content: " ▲"; font-size: 8px; color: var(--accent); }}
tr.down td.now::after {{ content: " ▼"; font-size: 8px; color: var(--accent); }}
tr.none td {{ color: var(--ink-3); font-family: var(--sans); font-size: 13.5px; }}
.more {{ margin-top: 10px; }}
.more summary {{ font-size: 12.5px; color: var(--ink-3); cursor: pointer; }}
.more summary:hover {{ color: var(--ink); }}
.more table {{ margin-top: 8px; }}

.card[hidden] {{ display: none; }}
@media (max-width: 900px) {{
  .body {{ grid-template-columns: 1fr; gap: 18px; }}
  audio {{ max-width: 100%; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
</head><body>
<div class="wrap">

<header class="top">
  <h1>Sonic Logo Stimuli</h1>
  <p class="sub">Every stimulus pair produced by the generation stage. Each card holds the
  brand brief that seeded it, both versions to listen to, where the two estimators placed
  them, and which synthesiser parameters the optimiser changed.</p>
  <div class="stats">
    <div><b>{len(runs)}</b><span>pairs</span></div>
    <div><b>{len(runs) * 2}</b><span>stimuli</span></div>
    <div><b>{held}/{len(runs)}</b><span>stayed in quadrant</span></div>
    <div><b>{tied}/{len(runs)}</b><span>unchanged pairs</span></div>
  </div>
</header>

<section class="roles">
  <h2>Which estimator is which</h2>
  <div class="rgrid">
    {role_cards}
  </div>
  <p class="convention">A trailing <b>2</b> marks an estimator chosen by the architecture
  Coach and judge are trained on different corpora and are never the same model, so
  agreement between them carries information. Where an estimator was replaced during the
  study, only the one actually used is shown; the filename beside each name above is the
  artefact it came from. <b>First candidate</b> is iteration 0 of a run, before
  any revision; <b>best candidate</b> is the best the loop found within its iteration cap.
  Both come from the same run and the same brief.</p>
</section>

<section class="legend">
  <h2>Reading the plots</h2>
  <p>Valence runs left to right, arousal bottom to top. The dashed ring is where the brief
  aimed. The hollow dot is the first candidate, the filled dot is the best one the optimiser
  found, and the line between them is how far it travelled. {scale_note} No point is ever
  clipped, and the numbers under each plot are the coordinates being drawn, so the picture
  can be checked against the CSV. The quadrant badge on each card reports {coach_label}'s
  verdict, since that is the estimator the stopping rule could act on; a held-out judge may
  place the same stimulus on the other side of an axis.</p>
  <div class="keys">
    <span class="key"><span class="swatch sw-t"></span>brief target</span>
    <span class="key"><span class="swatch sw-n"></span>first candidate</span>
    <span class="key"><span class="swatch sw-o"></span>best candidate</span>
    <span class="key"><b>{coach_label}</b>&nbsp;guided the search</span>
    {"".join(f'<span class="key"><b>{lab}</b>&nbsp;held out</span>' for _n, lab, d in judges if d)}
  </div>
</section>

<div class="filters">
  <span class="lab">Quadrant</span>
  {quad_btns}
  <span class="lab" style="margin-left:14px">Show only</span>
  <button class="chip" data-f="held" data-v="1">held quadrant</button>
  <button class="chip" data-f="held" data-v="0">crossed the axis</button>
  <button class="chip" data-f="tied" data-v="1">unchanged pairs</button>
  <span class="count" id="count"></span>
</div>

{cards}

</div>
<script>
(function () {{
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chip'));
  var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));
  var count = document.getElementById('count');
  var active = {{}};

  function apply() {{
    var shown = 0;
    cards.forEach(function (c) {{
      var ok = Object.keys(active).every(function (k) {{
        return !active[k] || c.dataset[k] === active[k];
      }});
      c.hidden = !ok;
      if (ok) shown++;
    }});
    count.textContent = shown + (shown === 1 ? ' pair' : ' pairs');
  }}

  chips.forEach(function (chip) {{
    chip.setAttribute('aria-pressed', 'false');
    chip.addEventListener('click', function () {{
      var f = chip.dataset.f, v = chip.dataset.v;
      var on = active[f] === v;
      chips.forEach(function (o) {{
        if (o.dataset.f === f) o.setAttribute('aria-pressed', 'false');
      }});
      active[f] = on ? null : v;
      chip.setAttribute('aria-pressed', on ? 'false' : 'true');
      apply();
    }});
  }});

  // One player at a time, so comparisons stay honest.
  var players = Array.prototype.slice.call(document.querySelectorAll('audio'));
  players.forEach(function (p) {{
    p.addEventListener('play', function () {{
      players.forEach(function (o) {{ if (o !== p) o.pause(); }});
    }});
  }});

  apply();
}})();
</script>
</body></html>'''

    out = MP3_DIR / "index.html"
    MP3_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    names = [lab for _n, lab, d in judges if d]
    print(f"Wrote {out}  ({len(runs)} pairs, {len(runs)*2} stimuli)")
    print(f"coach: {coach_label}  (models/{coach})")
    for n, lab, d in judges:
        if d:
            print(f"judge: {lab}  (models/{n})")
    hidden = ["estimator_B" if f.stem == "h1_estimator_b"
              else f.stem.replace("h1_estimator_b_", "")
              for f in sorted(ANALYSIS_DIR.glob("h1_estimator_b*.csv"))]
    shown = {coach} | {n for n, _l, d in judges if d}
    skipped = [h for h in hidden if h not in shown]
    if skipped and not only:
        print(f"not plotted (superseded or the coach re-scored): {', '.join(skipped)}"
              f"   -- use --all to include them")
    if scale == "shared":
        print(f"plot axes: shared, +/-{max(doms.values()):.2f}")
    else:
        print("plot axes: per estimator -> " +
              ", ".join(f"{k} +/-{v:.2f}" for k, v in doms.items()))
    if not names:
        print("note: no h1_estimator_b*.csv found in data/analysis/ — judge plots omitted.")
        print("      Run: python src/analysis/score_estimator_b.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="shared", choices=["shared", "per-judge"],
                    help="'shared' puts every estimator on one axis so compression is "
                         "visible; 'per-judge' scales each to its own range so individual "
                         "positions are readable")
    ap.add_argument("--all", dest="show_all", action="store_true",
                    help="also plot superseded estimators (by default only the "
                         "latest version of each is shown)")
    ap.add_argument("--judges", default=None,
                    help="Comma-separated estimator names to plot, in order. Default: "
                         "every estimator scored in data/analysis/ "
                         "(e.g. estimator_B,estimator_B2)")
    a = ap.parse_args()
    build([j.strip() for j in a.judges.split(",")] if a.judges else None,
          a.scale, a.show_all)