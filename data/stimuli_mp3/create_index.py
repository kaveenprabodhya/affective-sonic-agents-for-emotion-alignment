#!/usr/bin/env python3
"""Build the stimuli listening page.

Reads the generation manifest, the brand briefs and (if present) the held-out
Estimator B scores, then writes a single self-contained index.html next to the
MP3s.

The page is organised as:
    quadrant -> brief -> repeated generation run

Each brief therefore appears once. Its three repeated runs are exposed through
run-selector buttons rather than being rendered as three consecutive full cards.
The first quadrant (HV / HA) is selected by default so the page never opens with
all 48 pairs visible at once.

Run from the project root:
    python data/stimuli_mp3/create_index.py
    python data/stimuli_mp3/create_index.py --judges estimator_B2
"""

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MP3_DIR = ROOT / "data" / "stimuli_mp3"
MANIFEST = ROOT / "data" / "stimuli" / "manifest.json"
BRIEFS = ROOT / "config" / "briefs.yaml"
ANALYSIS_DIR = ROOT / "data" / "analysis"

# Keep this order everywhere on the page.
# HV_HA is intentionally first because it is the default active filter.
QUAD_ORDER = ["HV_HA", "HV_LA", "LV_HA", "LV_LA"]

# Quadrant hues follow the circumplex: warm where arousal is high, cool where it
# is low; saturated where valence is negative. Colour encodes position, not decoration.
QUAD = {
    "HV_HA": ("#C2621B", "High valence · High arousal", "excited, energetic"),
    "LV_HA": ("#A62F47", "Low valence · High arousal", "tense, agitated"),
    "LV_LA": ("#2E5E8E", "Low valence · Low arousal", "subdued, melancholy"),
    "HV_LA": ("#3D7A5A", "High valence · Low arousal", "calm, contented"),
}

# Parameters shown in this order; anything not listed still renders, after these.
PARAM_ORDER = [
    "tempo_bpm",
    "notes_per_beat",
    "instrument",
    "mode",
    "contour",
    "pitch_center_midi",
    "pitch_range",
    "dynamics",
    "articulation",
]

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

PARAM_UNIT = {
    "tempo_bpm": " BPM",
    "pitch_center_midi": " MIDI",
    "pitch_range": " st",
}

# Plot domain. Computed from the data at build time so no point is ever clamped
# to the frame edge. All plots share one scale by default.
PLOT_PX = 168
DOMAIN_MIN = 0.2
DOMAIN_HEADROOM = 1.1


def esc(value):
    """HTML-escape values inserted into generated markup."""
    return html.escape(str(value), quote=True)


def load_briefs(path):
    """Minimal YAML reader for the flat brief structure.

    PyYAML is used when available; the fallback keeps the page buildable from a
    bare checkout.
    """
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

            cur = {
                "id": line.split(":", 1)[1].strip().strip('"\''),
                "target": {},
            }

        elif cur is None:
            continue

        elif line.startswith("quadrant:"):
            cur["quadrant"] = (
                line.split(":", 1)[1]
                .strip()
                .strip('"\'')
            )

        elif line.startswith("brand_description:"):
            cur["brand_description"] = (
                line.split(":", 1)[1]
                .strip()
                .strip('"\'')
            )

        elif line.startswith("valence:"):
            cur["target"]["valence"] = float(
                line.split(":", 1)[1]
            )

        elif line.startswith("arousal:"):
            cur["target"]["arousal"] = float(
                line.split(":", 1)[1]
            )

    if cur:
        briefs[cur["id"]] = cur

    return briefs


def split_name(name):
    """estimator_B2 -> ("B", 2); estimator_B -> ("B", 1)."""
    m = re.match(r"^estimator_([A-Za-z]+)(\d*)$", name)

    if not m:
        return name, 1

    return m.group(1), int(m.group(2) or 1)


def display_name(name):
    """Human-facing estimator label.

    The trailing model version is an implementation detail; the artefact name is
    still shown in the estimator-role panel for traceability.
    """
    base, _ = split_name(name)
    return f"Estimator {base}"


def keep_final(found):
    """Among estimators sharing a base letter, keep only the highest version."""
    best = {}

    for name, path in found:
        base, ver = split_name(name)

        if base not in best or ver > split_name(best[base][0])[1]:
            best[base] = (name, path)

    order = [n for n, _ in found]

    return sorted(
        best.values(),
        key=lambda t: order.index(t[0]),
    )


def discover_judges(only=None, coach=None, show_all=False):
    """Find every held-out judge that has been scored, in a stable order."""
    found = []

    inc = ANALYSIS_DIR / "h1_estimator_b.csv"

    if inc.exists():
        found.append(("estimator_B", inc))

    for f in sorted(ANALYSIS_DIR.glob("h1_estimator_b_*.csv")):
        found.append(
            (
                f.stem.replace("h1_estimator_b_", ""),
                f,
            )
        )

    if only:
        found = [
            (n, f)
            for n, f in found
            if n in only
        ]

    else:
        if not show_all:
            found = keep_final(found)

        if coach:
            found = [
                (n, f)
                for n, f in found
                if n != coach
            ]

    return [
        (
            n,
            display_name(n),
            load_est_b(f),
        )
        for n, f in found
    ]


def load_est_b(path):
    if not path.exists():
        return {}

    out = {}

    for row in csv.DictReader(path.open()):
        out[(row["brief"], int(row["run"]))] = {
            "nonopt": (
                float(row["nonopt_B_v"]),
                float(row["nonopt_B_a"]),
            ),
            "opt": (
                float(row["opt_B_v"]),
                float(row["opt_B_a"]),
            ),
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


def plot(
    target,
    nonopt,
    opt,
    colour,
    dom,
    marker_id,
    size=168,
    pad=14,
):
    """Draw target, first candidate and best candidate on a VA plane."""
    tx, ty = to_xy(
        *target,
        dom=dom,
        size=size,
        pad=pad,
    )

    nx, ny = to_xy(
        *nonopt,
        dom=dom,
        size=size,
        pad=pad,
    )

    ox, oy = to_xy(
        *opt,
        dom=dom,
        size=size,
        pad=pad,
    )

    mid = size / 2

    moved = (
        abs(nx - ox) > 1.5
        or abs(ny - oy) > 1.5
    )

    arrow = ""

    if moved:
        arrow = (
            f'<line class="mv" '
            f'x1="{nx}" y1="{ny}" '
            f'x2="{ox}" y2="{oy}" '
            f'marker-end="url(#{marker_id})"/>'
        )

    return f'''
<svg
    class="plot"
    viewBox="0 0 {size} {size}"
    role="img"
    aria-label="Valence-arousal plot: target, first candidate, best candidate"
>
    <defs>
        <marker
            id="{marker_id}"
            markerWidth="7"
            markerHeight="7"
            refX="5.5"
            refY="3"
            orient="auto"
        >
            <path
                d="M0,0 L6,3 L0,6 z"
                fill="{colour}"
            />
        </marker>
    </defs>

    <rect
        x="{pad}"
        y="{pad}"
        width="{size - 2 * pad}"
        height="{size - 2 * pad}"
        class="frame"
    />

    <line
        x1="{mid}"
        y1="{pad}"
        x2="{mid}"
        y2="{size - pad}"
        class="axis"
    />

    <line
        x1="{pad}"
        y1="{mid}"
        x2="{size - pad}"
        y2="{mid}"
        class="axis"
    />

    <circle
        cx="{tx}"
        cy="{ty}"
        r="7"
        class="target"
        stroke="{colour}"
    />

    <circle
        cx="{tx}"
        cy="{ty}"
        r="1.6"
        fill="{colour}"
    />

    {arrow}

    <circle
        cx="{nx}"
        cy="{ny}"
        r="4.5"
        class="dot-non"
    />

    <circle
        cx="{ox}"
        cy="{oy}"
        r="4.5"
        class="dot-opt"
        fill="{colour}"
    />
</svg>
'''


def xy(p):
    return f"{p[0]:+.2f},{p[1]:+.2f}"


def pcol(
    title,
    sub,
    target,
    non,
    opt,
    colour,
    size,
    dom,
    marker_id,
):
    """One plot column: heading, plot, and coordinates."""
    return (
        '<div class="pcol">'
        f'<h4>{esc(title)} '
        f'<small>{esc(sub)} &middot; &plusmn;{dom:.2f}</small>'
        '</h4>'
        f'{plot(target, non, opt, colour, dom, marker_id, size=size)}'
        f'<div class="coords">'
        f'{xy(non)} <i>&rarr;</i> {xy(opt)}'
        '</div>'
        '</div>'
    )


def fmt(key, val):
    return f"{val}{PARAM_UNIT.get(key, '')}"


def param_rows(p_non, p_opt):
    keys = [
        k
        for k in PARAM_ORDER
        if k in p_non or k in p_opt
    ]

    keys += [
        k
        for k in sorted(set(p_non) | set(p_opt))
        if k not in keys
    ]

    changed = []
    unchanged = []

    for k in keys:
        a = p_non.get(k)
        b = p_opt.get(k)

        label = PARAM_LABEL.get(
            k,
            k.replace("_", " ").capitalize(),
        )

        if a != b:
            direction = ""

            if (
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
            ):
                direction = " up" if b > a else " down"

            changed.append(
                f'<tr class="chg{direction}">'
                f'<th>{esc(label)}</th>'
                f'<td class="was">{esc(fmt(k, a))}</td>'
                '<td class="arrow" aria-hidden="true">&rarr;</td>'
                f'<td class="now">{esc(fmt(k, b))}</td>'
                '</tr>'
            )

        else:
            unchanged.append(
                '<tr>'
                f'<th>{esc(label)}</th>'
                f'<td colspan="3">{esc(fmt(k, a))}</td>'
                '</tr>'
            )

    return changed, unchanged


def distance_status(dn, do):
    """Return label/class/delta for one distance change."""
    delta = dn - do

    if delta > 0.0005:
        return "closer", "good", delta

    if delta < -0.0005:
        return "further", "bad", delta

    return "no change", "flat", delta


def dist_pair(label, dn, do):
    if dn is None or do is None:
        return ""

    status, cls, _ = distance_status(dn, do)

    return (
        '<div class="dist">'
        f'<span class="dl">{esc(label)}</span>'
        f'<span class="dv">'
        f'{dn:.3f} <i>&rarr;</i> {do:.3f}'
        '</span>'
        f'<span class="dd {cls}">{status}</span>'
        '</div>'
    )


def run_state(run, target):
    non = run["non_optimised"]
    opt = run["optimised"]

    n_est = tuple(non["est"])
    o_est = tuple(opt["est"])

    tied = (
        non["file"] == opt["file"]
        or (
            n_est == o_est
            and non["distance"] == opt["distance"]
        )
    )

    held = opt.get("quadrant_ok")

    if held is None:
        held = same_sign(o_est, target)

    return tied, held


def summary_measure(run, judges):
    """Choose the distance shown on the compact run selector.

    Prefer the first available held-out judge because the selector is intended
    to summarise independent evaluation. If no held-out score exists, fall back
    to the optimisation coach.
    """
    key = (
        run["brief"],
        run["run"],
    )

    for _name, label, data in judges:
        d = data.get(key)

        if d:
            return (
                label,
                d["nonopt_dist"],
                d["opt_dist"],
                True,
            )

    non = run["non_optimised"]
    opt = run["optimised"]

    return (
        "Coach",
        non["distance"],
        opt["distance"],
        False,
    )


def run_button(run, judges, selected=False):
    label, dn, do, held_out = summary_measure(
        run,
        judges,
    )

    status, cls, _ = distance_status(
        dn,
        do,
    )

    if held_out:
        metric_label = f"{label} · held out"
    else:
        metric_label = label

    return f'''
<button
    type="button"
    class="run-tab {cls}"
    data-run-tab="{run['run']}"
    aria-selected="{'true' if selected else 'false'}"
>
    <span class="rt-top">
        Run {run['run']}
    </span>

    <span class="rt-status">
        {status}
    </span>

    <span class="rt-dist">
        {dn:.3f} &rarr; {do:.3f}
    </span>

    <span class="rt-source">
        {esc(metric_label)}
    </span>
</button>
'''


def run_panel(
    run,
    brief,
    judges,
    doms,
    coach_label,
    panel_uid,
    selected=False,
):
    bid = run["brief"]
    ridx = run["run"]

    colour, _quad_name, _quad_feel = QUAD.get(
        brief.get("quadrant", ""),
        ("#555", "", ""),
    )

    target = tuple(run["target"])

    non = run["non_optimised"]
    opt = run["optimised"]

    n_est = tuple(non["est"])
    o_est = tuple(opt["est"])

    tied, held = run_state(
        run,
        target,
    )

    changed, unchanged = param_rows(
        non["params"],
        opt["params"],
    )

    jrows = [
        (
            name,
            label,
            data.get((bid, ridx)),
        )
        for name, label, data in judges
    ]

    badges = []

    if tied:
        badges.append(
            '<span class="badge tie">'
            'no change — best was the first candidate'
            '</span>'
        )

    badges.append(
        f'<span class="badge {"held" if held else "crossed"}">'
        f'{"held quadrant" if held else "crossed the axis"} '
        f'<em>per {esc(coach_label)}</em>'
        '</span>'
    )

    if run.get("reached_threshold"):
        badges.append(
            '<span class="badge met">'
            'met both stopping criteria'
            '</span>'
        )

    changed_html = (
        "".join(changed)
        or (
            '<tr class="none">'
            '<td colspan="4">'
            'No parameters changed.'
            '</td>'
            '</tr>'
        )
    )

    more = ""

    if unchanged:
        more = (
            '<details class="more">'
            f'<summary>'
            f'Unchanged parameters ({len(unchanged)})'
            '</summary>'
            '<table class="params">'
            f'{"".join(unchanged)}'
            '</table>'
            '</details>'
        )

    judge_plots = "".join(
        pcol(
            label,
            "held out",
            target,
            d["nonopt"],
            d["opt"],
            colour,
            PLOT_PX,
            doms[label],
            (
                f"ah-{panel_uid}-"
                f"{re.sub(r'[^A-Za-z0-9_-]', '', name)}"
            ),
        )
        for name, label, d in jrows
        if d
    )

    judge_dists = "".join(
        dist_pair(
            f"{label} — independent",
            d["nonopt_dist"],
            d["opt_dist"],
        )
        for _name, label, d in jrows
        if d
    )

    non_src = esc(
        f"{Path(non['file']).stem}.mp3"
    )

    opt_src = esc(
        f"{Path(opt['file']).stem}.mp3"
    )

    hidden_attr = (
        ""
        if selected
        else "hidden"
    )

    coach_plot = pcol(
        coach_label,
        "coach",
        target,
        n_est,
        o_est,
        colour,
        PLOT_PX,
        doms[coach_label],
        f"ah-{panel_uid}-coach",
    )

    coach_dist = dist_pair(
        f"{coach_label} — coach",
        non["distance"],
        opt["distance"],
    )

    return f'''
<section
    class="run-panel"
    data-run-panel="{ridx}"
    {hidden_attr}
>
    <div class="run-head">
        <div>
            <span class="eyebrow">
                Selected repetition
            </span>

            <h4>
                Run {ridx} details
            </h4>
        </div>

        <div class="badges">
            {''.join(badges)}
        </div>
    </div>

    <div class="body">
        <div class="left">

            <div class="listen">

                <div class="track">
                    <div class="tlab">
                        <strong>
                            First candidate
                        </strong>

                        <span>
                            non-optimised · iteration 0
                        </span>
                    </div>

                    <audio
                        controls
                        preload="none"
                        src="{non_src}"
                    ></audio>
                </div>

                <div class="track">
                    <div class="tlab">
                        <strong>
                            Best candidate
                        </strong>

                        <span>
                            optimised · iteration
                            {opt.get('iteration', 0)}
                            of {run['iterations']}
                        </span>
                    </div>

                    <audio
                        controls
                        preload="none"
                        src="{opt_src}"
                    ></audio>
                </div>

            </div>

            <div class="dists">
                {coach_dist}
                {judge_dists}
            </div>

        </div>

        <div class="plots">
            {coach_plot}
            {judge_plots}
        </div>

    </div>

    <div class="whatchanged">

        <h4>
            What changed in the synthesiser
        </h4>

        <table class="params">
            {changed_html}
        </table>

        {more}

    </div>

</section>
'''


def brief_card(
    brief_runs,
    brief,
    judges,
    idx,
    doms,
    coach_label,
):
    """Render one brief once, with all repeated runs nested inside it."""
    brief_runs = sorted(
        brief_runs,
        key=lambda r: r["run"],
    )

    first = brief_runs[0]

    bid = first["brief"]

    quadrant = brief.get(
        "quadrant",
        "",
    )

    colour, quad_name, quad_feel = QUAD.get(
        quadrant,
        ("#555", "", ""),
    )

    target = tuple(
        first["target"]
    )

    states = [
        run_state(
            r,
            tuple(r["target"]),
        )
        for r in brief_runs
    ]

    any_tied = any(
        tied
        for tied, _held in states
    )

    any_held = any(
        held
        for _tied, held in states
    )

    any_crossed = any(
        not held
        for _tied, held in states
    )

    run_tabs = "".join(
        run_button(
            r,
            judges,
            selected=(i == 0),
        )
        for i, r in enumerate(brief_runs)
    )

    panels = "".join(
        run_panel(
            r,
            brief,
            judges,
            doms,
            coach_label,
            f"{bid}-r{r['run']}",
            selected=(i == 0),
        )
        for i, r in enumerate(brief_runs)
    )

    return f'''
<article
    class="brief-card"
    data-quad="{esc(quadrant)}"
    data-any-held="{'1' if any_held else '0'}"
    data-any-crossed="{'1' if any_crossed else '0'}"
    data-any-tied="{'1' if any_tied else '0'}"
    data-run-count="{len(brief_runs)}"
    style="--accent:{colour}"
>

    <header class="brief-head">

        <div class="brief-title">

            <span class="brief-index">
                {idx:02d}
            </span>

            <div>

                <h3>
                    {esc(bid)}
                </h3>

                <div class="brief-meta">

                    <span class="quad">
                        {esc(quad_name)}
                        <em>
                            {esc(quad_feel)}
                        </em>
                    </span>

                    <span class="tgt">
                        target {xy(target)}
                    </span>

                </div>

            </div>

        </div>

        <span class="repeat-count">
            {len(brief_runs)} repeated runs
        </span>

    </header>

    <p class="brief">
        {esc(brief.get('brand_description', '—'))}
    </p>

    <div class="run-selector">

        <div class="run-selector-head">

            <div>

                <span class="eyebrow">
                    Repeated generation
                </span>

                <h4>
                    Compare the three independent runs
                </h4>

            </div>

            <p>
                Choose a run to hear its pair and inspect
                its estimator movement.
            </p>

        </div>

        <div
            class="run-tabs"
            role="tablist"
            aria-label="{esc(bid)} generation runs"
        >
            {run_tabs}
        </div>

    </div>

    <div class="run-panels">
        {panels}
    </div>

</article>
'''


def build(
    only=None,
    scale="shared",
    show_all=False,
):
    if not MANIFEST.exists():
        sys.exit(
            f"No manifest at {MANIFEST}. "
            "Run the generation stage first."
        )

    runs = json.loads(
        MANIFEST.read_text()
    )

    briefs = load_briefs(
        BRIEFS
    )

    # The coach that actually produced these estimates.
    # Recorded per run by run_generation.py;
    # older manifests predate that field.
    coach = next(
        (
            r["coach"]
            for r in runs
            if r.get("coach")
        ),
        "estimator_A",
    )

    coach_label = display_name(
        coach
    )

    judges = discover_judges(
        only,
        coach,
        show_all,
    )

    def dom_of(vals):
        m = max(
            (
                abs(v)
                for v in vals
            ),
            default=DOMAIN_MIN,
        )

        return max(
            DOMAIN_MIN,
            round(
                m * DOMAIN_HEADROOM + 0.049,
                1,
            ),
        )

    tgt = [
        c
        for r in runs
        for c in r["target"]
    ]

    per = {
        coach_label:
        tgt
        + [
            c
            for r in runs
            for c in (
                list(
                    r["non_optimised"]["est"]
                )
                + list(
                    r["optimised"]["est"]
                )
            )
        ]
    }

    for _name, label, data in judges:
        per[label] = (
            tgt
            + [
                c
                for d in data.values()
                for c in (
                    list(d["nonopt"])
                    + list(d["opt"])
                )
            ]
        )

    if scale == "shared":
        one = dom_of(
            [
                c
                for values in per.values()
                for c in values
            ]
        )

        doms = {
            k: one
            for k in per
        }

    else:
        doms = {
            k: dom_of(v)
            for k, v in per.items()
        }

    missing = [
        r
        for r in runs
        if not (
            MP3_DIR
            / (
                f"{Path(r['optimised']['file']).stem}.mp3"
            )
        ).exists()
    ]

    if missing:
        print(
            f"warning: {len(missing)} pairs have no MP3 yet "
            "— run convert_2_mp3.sh"
        )

    runs.sort(
        key=lambda r: (
            r["brief"],
            r["run"],
        )
    )

    # ------------------------------------------------------------
    # GROUP RUNS BY BRIEF
    # ------------------------------------------------------------

    by_brief = defaultdict(list)

    for r in runs:
        by_brief[r["brief"]].append(r)

    def brief_sort_key(bid):
        b = briefs.get(
            bid,
            {},
        )

        q = b.get(
            "quadrant",
            "",
        )

        if q in QUAD_ORDER:
            q_index = QUAD_ORDER.index(q)
        else:
            q_index = len(QUAD_ORDER)

        return (
            q_index,
            bid,
        )

    ordered_briefs = sorted(
        by_brief,
        key=brief_sort_key,
    )

    brief_cards = "\n".join(
        brief_card(
            by_brief[bid],
            briefs.get(
                bid,
                {},
            ),
            judges,
            i,
            doms,
            coach_label,
        )
        for i, bid in enumerate(
            ordered_briefs,
            1,
        )
    )

    held = 0
    tied = 0

    for r in runs:
        target = tuple(
            r["target"]
        )

        is_tied, is_held = run_state(
            r,
            target,
        )

        held += int(is_held)
        tied += int(is_tied)

    def role_card(
        name,
        label,
        kind,
    ):
        artefact = (
            f' <span class="artefact">'
            f'models/{esc(name)}'
            '</span>'
        )

        if kind == "coach":
            tag = "coach"

            body = (
                "Guides the optimisation loop. "
                "It scores every candidate, reports the gap "
                "to the target back to the language model, "
                "and decides when the loop stops. "
                "It sees everything, so agreeing with it is "
                "not evidence that a logo carries the intended emotion."
            )

        elif name == "estimator_B":
            tag = "held out"

            body = (
                "The judge specified before the study ran. "
                "Never consulted during generation: it scores "
                "the finished audio only, so its verdict is "
                "independent of the search."
            )

        else:
            tag = "held out"

            body = (
                "An additional held-out judge, chosen by the "
                "architecture comparison in models/selection/. "
                "Never consulted during generation."
            )

        return (
            '<div class="role">'
            f'<h3>{esc(label)} '
            f'<span class="tag {tag.replace(" ", "")}">'
            f'{tag}'
            '</span>'
            f'{artefact}'
            '</h3>'
            f'<p>{body}</p>'
            '</div>'
        )

    role_cards = (
        role_card(
            coach,
            coach_label,
            "coach",
        )
        + "".join(
            role_card(
                name,
                label,
                "judge",
            )
            for name, label, data in judges
            if data
        )
    )

    if scale == "shared":
        scale_note = (
            f"Every plot shares one scale, "
            f"&plusmn;{max(doms.values()):.2f} "
            "of the &plusmn;1 circumplex, so a judge "
            "whose dots barely move really is barely moving."
        )

    else:
        scale_note = (
            "Each estimator is scaled to its own range — "
            "read the axis label before comparing columns, "
            "because the columns are not on the same scale."
        )

    available_quads = {
        b.get("quadrant")
        for b in briefs.values()
        if b.get("quadrant")
    }

    quads = [
        q
        for q in QUAD_ORDER
        if q in available_quads
    ]

    quads += sorted(
        available_quads - set(quads)
    )

    quad_btns = "".join(
        (
            f'<button '
            f'type="button" '
            f'class="chip" '
            f'data-f="quad" '
            f'data-v="{esc(q)}" '
            f'style="--c:{QUAD.get(q, ("#555",))[0]}">'
            f'{esc(q.replace("_", " / "))}'
            '</button>'
        )
        for q in quads
    )

    if "HV_HA" in quads:
        default_quad = "HV_HA"

    elif quads:
        default_quad = quads[0]

    else:
        default_quad = ""

    judge_keys_html = "".join(
        (
            '<span class="key">'
            f'<b>{esc(label)}</b>'
            '&nbsp;held out'
            '</span>'
        )
        for _name, label, data in judges
        if data
    )

    html_doc = f'''
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
    Sonic Logo Stimuli
</title>

<link
    rel="preconnect"
    href="https://fonts.googleapis.com"
>

<link
    rel="preconnect"
    href="https://fonts.gstatic.com"
    crossorigin
>

<link
    href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap"
    rel="stylesheet"
>

<style>

:root {{
    --ink: #14181D;
    --ink-2: #4A5560;
    --ink-3: #79858F;

    --ground: #FAFBFC;
    --panel: #FFFFFF;

    --line: #E1E6EA;
    --line-2: #EDF1F4;

    --good: #2E7D5B;
    --bad: #B03A3A;

    --sans:
        'Space Grotesk',
        system-ui,
        -apple-system,
        sans-serif;

    --mono:
        'IBM Plex Mono',
        ui-monospace,
        'SF Mono',
        Menlo,
        monospace;
}}


* {{
    box-sizing: border-box;
}}


html {{
    scroll-behavior: smooth;
}}


body {{
    margin: 0;

    background: var(--ground);
    color: var(--ink);

    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.5;

    -webkit-font-smoothing: antialiased;
}}


.wrap {{
    max-width: 1120px;
    margin: 0 auto;
    padding: 0 24px 96px;
}}


/* ============================================================
   TOP
   ============================================================ */

header.top {{
    padding: 56px 0 28px;

    border-bottom: 2px solid var(--ink);

    margin-bottom: 28px;
}}


h1 {{
    font-size: clamp(
        30px,
        5vw,
        46px
    );

    font-weight: 700;

    letter-spacing: -0.025em;

    margin: 0 0 10px;
}}


.sub {{
    color: var(--ink-2);

    max-width: 68ch;

    margin: 0 0 22px;
}}


.stats {{
    display: flex;
    flex-wrap: wrap;

    gap: 28px;

    font-family: var(--mono);
    font-size: 13px;
}}


.stats div {{
    display: flex;
    flex-direction: column;

    gap: 2px;
}}


.stats b {{
    font-size: 21px;
    font-weight: 500;
}}


.stats span {{
    color: var(--ink-3);

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.08em;
}}


/* ============================================================
   ESTIMATOR ROLES
   ============================================================ */

.roles {{
    background: var(--panel);

    border: 1px solid var(--line);

    border-radius: 10px;

    padding: 20px 22px;

    margin-bottom: 18px;
}}


.roles h2 {{
    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.1em;

    color: var(--ink-3);

    margin: 0 0 14px;

    font-weight: 500;
}}


.rgrid {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(240px, 1fr)
        );

    gap: 18px;
}}


.role h3 {{
    margin: 0 0 5px;

    font-size: 15px;

    font-weight: 700;

    display: flex;

    align-items: baseline;

    gap: 8px;

    flex-wrap: wrap;
}}


.role p {{
    margin: 0;

    font-size: 13.5px;

    color: var(--ink-2);
}}


.tag {{
    font-family: var(--mono);

    font-size: 10px;

    font-weight: 400;

    padding: 2px 7px;

    border-radius: 4px;

    text-transform: uppercase;

    letter-spacing: 0.06em;
}}


.tag.coach {{
    background: #FDF0E4;

    color: #A05418;

    border: 1px solid #F0D8BF;
}}


.tag.heldout {{
    background: #EAF1F8;

    color: #2E5E8E;

    border: 1px solid #CEDEEE;
}}


.artefact {{
    font-family: var(--mono);

    font-size: 10.5px;

    font-weight: 400;

    color: var(--ink-3);
}}


.convention {{
    margin: 16px 0 0;

    padding-top: 13px;

    border-top: 1px solid var(--line-2);

    font-size: 13.5px;

    color: var(--ink-2);
}}


/* ============================================================
   LEGEND
   ============================================================ */

.legend {{
    display: grid;

    grid-template-columns:
        auto
        1fr;

    gap: 18px 22px;

    background: var(--panel);

    border: 1px solid var(--line);

    border-radius: 10px;

    padding: 20px 22px;

    margin-bottom: 26px;

    align-items: center;
}}


.legend h2 {{
    grid-column: 1 / -1;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.1em;

    color: var(--ink-3);

    margin: 0;

    font-weight: 500;
}}


.legend p {{
    margin: 0;

    color: var(--ink-2);

    font-size: 14px;
}}


.legend .keys {{
    grid-column: 1 / -1;

    display: flex;

    flex-wrap: wrap;

    gap: 20px;

    font-size: 13px;

    color: var(--ink-2);

    padding-top: 4px;

    border-top: 1px solid var(--line-2);
}}


.key {{
    display: flex;

    align-items: center;

    gap: 7px;
}}


.swatch {{
    width: 13px;
    height: 13px;

    border-radius: 50%;

    flex: none;
}}


.sw-t {{
    border: 2px solid var(--ink-2);

    background: none;
}}


.sw-n {{
    background: none;

    border: 2px solid var(--ink-3);
}}


.sw-o {{
    background: var(--ink);
}}


/* ============================================================
   FILTER BAR
   ============================================================ */

.filters {{
    display: flex;

    flex-wrap: wrap;

    gap: 8px;

    align-items: center;

    margin-bottom: 24px;

    position: sticky;

    top: 0;

    background: rgba(
        250,
        251,
        252,
        0.96
    );

    backdrop-filter: blur(8px);

    padding: 12px 0;

    z-index: 5;

    border-bottom: 1px solid var(--line);
}}


.filters .lab {{
    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.08em;

    color: var(--ink-3);

    margin-right: 4px;
}}


.chip {{
    font:
        500
        13px
        var(--sans);

    background: var(--panel);

    color: var(--ink-2);

    border: 1px solid var(--line);

    border-radius: 999px;

    padding: 7px 14px;

    cursor: pointer;
}}


.chip:hover {{
    border-color: var(--ink-3);

    color: var(--ink);
}}


.chip[aria-pressed="true"] {{
    background: var(
        --c,
        var(--ink)
    );

    color: #fff;

    border-color: var(
        --c,
        var(--ink)
    );
}}


.chip:focus-visible,
.run-tab:focus-visible {{
    outline: 2px solid var(--ink);

    outline-offset: 2px;
}}


.count {{
    margin-left: auto;

    font-family: var(--mono);

    font-size: 12px;

    color: var(--ink-3);
}}


/* ============================================================
   BRIEF CARD
   ============================================================ */

.brief-card {{
    background: var(--panel);

    border: 1px solid var(--line);

    border-radius: 12px;

    padding: 24px;

    margin-bottom: 22px;

    border-left:
        4px
        solid
        var(--accent);
}}


.brief-head {{
    display: flex;

    justify-content: space-between;

    align-items: flex-start;

    gap: 16px;

    flex-wrap: wrap;
}}


.brief-title {{
    display: flex;

    align-items: flex-start;

    gap: 13px;

    min-width: 0;
}}


.brief-index {{
    font-family: var(--mono);

    font-size: 11px;

    color: var(--ink-3);

    margin-top: 5px;
}}


.brief-head h3 {{
    margin: 0;

    font-size: 22px;

    font-weight: 700;

    letter-spacing: -0.015em;
}}


.brief-meta {{
    display: flex;

    align-items: baseline;

    gap: 12px;

    flex-wrap: wrap;

    margin-top: 2px;
}}


.quad {{
    font-size: 12.5px;

    color: var(--accent);

    font-weight: 500;
}}


.quad em {{
    color: var(--ink-3);

    font-style: normal;

    font-weight: 400;
}}


.tgt {{
    font-family: var(--mono);

    font-size: 11.5px;

    color: var(--ink-3);
}}


.repeat-count {{
    font-family: var(--mono);

    font-size: 10.5px;

    text-transform: uppercase;

    letter-spacing: 0.06em;

    color: var(--ink-3);

    border: 1px solid var(--line);

    border-radius: 999px;

    padding: 5px 9px;
}}


.brief {{
    margin: 16px 0 20px;

    font-size: 15.5px;

    color: var(--ink-2);

    max-width: 78ch;

    padding-left: 14px;

    border-left: 2px solid var(--line);
}}


/* ============================================================
   RUN SELECTOR
   ============================================================ */

.run-selector {{
    border-top: 1px solid var(--line-2);

    border-bottom: 1px solid var(--line-2);

    padding: 17px 0 18px;

    margin-bottom: 20px;
}}


.run-selector-head {{
    display: flex;

    justify-content: space-between;

    gap: 18px;

    align-items: end;

    margin-bottom: 12px;
}}


.run-selector-head h4 {{
    margin: 1px 0 0;

    font-size: 15px;
}}


.run-selector-head p {{
    margin: 0;

    color: var(--ink-3);

    font-size: 12.5px;

    text-align: right;
}}


.eyebrow {{
    display: block;

    font-family: var(--mono);

    font-size: 9.5px;

    color: var(--ink-3);

    text-transform: uppercase;

    letter-spacing: 0.08em;
}}


.run-tabs {{
    display: grid;

    grid-template-columns:
        repeat(
            3,
            minmax(0, 1fr)
        );

    gap: 9px;
}}


.run-tab {{
    position: relative;

    display: grid;

    grid-template-columns:
        1fr
        auto;

    gap: 2px 10px;

    align-items: baseline;

    min-width: 0;

    text-align: left;

    background: #FBFCFD;

    color: var(--ink);

    border: 1px solid var(--line);

    border-radius: 9px;

    padding: 11px 12px 10px;

    cursor: pointer;

    font-family: var(--sans);
}}


.run-tab:hover {{
    border-color: var(--ink-3);
}}


.run-tab[aria-selected="true"] {{
    border-color: var(--accent);

    box-shadow:
        inset
        0
        0
        0
        1px
        var(--accent);

    background: #fff;
}}


.run-tab[aria-selected="true"]::before {{
    content: "";

    position: absolute;

    left: 12px;

    right: 12px;

    bottom: -1px;

    height: 2px;

    background: var(--accent);
}}


.rt-top {{
    font-size: 13px;

    font-weight: 700;
}}


.rt-status {{
    justify-self: end;

    font-family: var(--mono);

    font-size: 10.5px;

    border-radius: 3px;

    padding: 1px 6px;
}}


.run-tab.good .rt-status {{
    color: var(--good);

    background: #F1F9F4;
}}


.run-tab.bad .rt-status {{
    color: var(--bad);

    background: #FDF3F3;
}}


.run-tab.flat .rt-status {{
    color: var(--ink-3);

    background: var(--line-2);
}}


.rt-dist {{
    font-family: var(--mono);

    font-size: 11.5px;

    color: var(--ink-2);
}}


.rt-source {{
    grid-column: 1 / -1;

    font-family: var(--mono);

    font-size: 9.5px;

    color: var(--ink-3);

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}}


/* ============================================================
   RUN DETAIL PANEL
   ============================================================ */

.run-panel[hidden] {{
    display: none;
}}


.run-head {{
    display: flex;

    justify-content: space-between;

    gap: 16px;

    align-items: flex-start;

    flex-wrap: wrap;

    margin-bottom: 14px;
}}


.run-head h4 {{
    margin: 1px 0 0;

    font-size: 17px;
}}


.badges {{
    display: flex;

    gap: 6px;

    flex-wrap: wrap;
}}


.badge {{
    font-family: var(--mono);

    font-size: 10.5px;

    padding: 3px 9px;

    border-radius: 4px;

    border: 1px solid var(--line);

    color: var(--ink-2);
}}


.badge.held {{
    color: var(--good);

    border-color: #BFE0CE;

    background: #F1F9F4;
}}


.badge.crossed {{
    color: var(--bad);

    border-color: #EFC9C9;

    background: #FDF3F3;
}}


.badge em {{
    font-style: normal;

    opacity: 0.66;
}}


.badge.tie {{
    color: var(--ink-3);
}}


.badge.met {{
    color: var(--ink);

    border-color: var(--ink-3);
}}


/* ============================================================
   AUDIO + PLOTS
   ============================================================ */

.body {{
    display: grid;

    grid-template-columns:
        minmax(0, 1fr)
        auto;

    gap: 28px;

    align-items: start;

    margin-top: 4px;
}}


.left {{
    min-width: 0;
}}


.listen {{
    display: flex;

    flex-direction: column;

    gap: 12px;
}}


.track {{
    min-width: 0;
}}


.tlab {{
    display: flex;

    align-items: baseline;

    gap: 9px;

    margin-bottom: 5px;

    flex-wrap: wrap;
}}


.tlab strong {{
    font-size: 14px;
}}


.tlab span {{
    font-family: var(--mono);

    font-size: 11px;

    color: var(--ink-3);
}}


audio {{
    width: 100%;

    max-width: 420px;

    height: 36px;
}}


/* ============================================================
   DISTANCES
   ============================================================ */

.dists {{
    margin-top: 16px;

    border-top: 1px solid var(--line-2);

    padding-top: 12px;

    display: flex;

    flex-direction: column;

    gap: 7px;
}}


.dist {{
    display: flex;

    align-items: baseline;

    gap: 10px;

    flex-wrap: wrap;
}}


.dl {{
    font-size: 12.5px;

    color: var(--ink-2);

    min-width: 24ch;
}}


.dv {{
    font-family: var(--mono);

    font-size: 13px;
}}


.dv i {{
    color: var(--ink-3);

    font-style: normal;
}}


.dd {{
    font-family: var(--mono);

    font-size: 11px;

    padding: 1px 7px;

    border-radius: 3px;
}}


.dd.good {{
    color: var(--good);

    background: #F1F9F4;
}}


.dd.bad {{
    color: var(--bad);

    background: #FDF3F3;
}}


.dd.flat {{
    color: var(--ink-3);

    background: var(--line-2);
}}


/* ============================================================
   VA PLOTS
   ============================================================ */

.plots {{
    display: flex;

    gap: 16px;

    padding-bottom: 6px;

    max-width: 100%;

    overflow-x: auto;

    scroll-snap-type: x proximity;

    scrollbar-width: thin;

    scrollbar-color:
        var(--line)
        transparent;
}}


.plots::-webkit-scrollbar {{
    height: 7px;
}}


.plots::-webkit-scrollbar-track {{
    background: var(--line-2);

    border-radius: 4px;
}}


.plots::-webkit-scrollbar-thumb {{
    background: var(--line);

    border-radius: 4px;
}}


.plots::-webkit-scrollbar-thumb:hover {{
    background: var(--ink-3);
}}


.pcol {{
    flex: 0 0 auto;

    scroll-snap-align: start;
}}


.pcol h4 {{
    margin: 0 0 4px;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.07em;

    color: var(--ink-3);

    font-weight: 500;
}}


.pcol h4 small {{
    text-transform: none;

    letter-spacing: 0;

    font-size: 10.5px;

    color: var(--ink-3);

    opacity: 0.75;
}}


.plot {{
    width: 168px;

    height: 168px;

    display: block;
}}


.coords {{
    font-family: var(--mono);

    font-size: 10.5px;

    color: var(--ink-3);

    text-align: center;

    margin-top: 3px;
}}


.coords i {{
    font-style: normal;

    opacity: 0.6;
}}


.frame {{
    fill: #FCFDFD;

    stroke: var(--line);
}}


.axis {{
    stroke: var(--line);

    stroke-dasharray: 3 3;
}}


.target {{
    fill: none;

    stroke-width: 1.5;

    stroke-dasharray: 3 2.5;
}}


.dot-non {{
    fill: #fff;

    stroke: var(--ink-3);

    stroke-width: 2;
}}


.dot-opt {{
    stroke: #fff;

    stroke-width: 1.5;
}}


.mv {{
    stroke: var(--accent);

    stroke-width: 1.5;

    opacity: 0.65;
}}


/* ============================================================
   PARAMETERS
   ============================================================ */

.whatchanged {{
    margin-top: 20px;

    border-top: 1px solid var(--line-2);

    padding-top: 14px;
}}


.whatchanged h4 {{
    margin: 0 0 9px;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.07em;

    color: var(--ink-3);

    font-weight: 500;
}}


table.params {{
    border-collapse: collapse;

    font-size: 13.5px;
}}


table.params th {{
    text-align: left;

    font-weight: 400;

    color: var(--ink-2);

    padding: 3px 22px 3px 0;

    white-space: nowrap;
}}


table.params td {{
    font-family: var(--mono);

    font-size: 13px;

    padding: 3px 14px 3px 0;
}}


td.was {{
    color: var(--ink-3);
}}


td.arrow {{
    color: var(--ink-3);

    padding: 0 4px 0 0;
}}


td.now {{
    font-weight: 500;
}}


tr.up td.now::after {{
    content: " ▲";

    font-size: 8px;

    color: var(--accent);
}}


tr.down td.now::after {{
    content: " ▼";

    font-size: 8px;

    color: var(--accent);
}}


tr.none td {{
    color: var(--ink-3);

    font-family: var(--sans);

    font-size: 13.5px;
}}


.more {{
    margin-top: 10px;
}}


.more summary {{
    font-size: 12.5px;

    color: var(--ink-3);

    cursor: pointer;
}}


.more summary:hover {{
    color: var(--ink);
}}


.more table {{
    margin-top: 8px;
}}


/* ============================================================
   FILTER HIDING
   ============================================================ */

.brief-card[hidden] {{
    display: none;
}}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {{

    .body {{
        grid-template-columns: 1fr;

        gap: 18px;
    }}


    audio {{
        max-width: 100%;
    }}


    .run-selector-head {{
        align-items: start;
    }}

}}


@media (max-width: 680px) {{

    .wrap {{
        padding-left: 16px;

        padding-right: 16px;
    }}


    header.top {{
        padding-top: 38px;
    }}


    .filters {{
        gap: 6px;
    }}


    .filters .lab {{
        width: 100%;

        margin-top: 4px;
    }}


    .filters .lab:first-child {{
        margin-top: 0;
    }}


    .count {{
        width: 100%;

        margin-left: 0;

        margin-top: 4px;
    }}


    .brief-card {{
        padding: 18px 16px;
    }}


    .run-selector-head {{
        display: block;
    }}


    .run-selector-head p {{
        text-align: left;

        margin-top: 6px;
    }}


    .run-tabs {{
        grid-template-columns: 1fr;
    }}


    .run-tab {{
        grid-template-columns:
            auto
            1fr
            auto;

        align-items: center;
    }}


    .rt-status {{
        justify-self: start;
    }}


    .rt-dist {{
        justify-self: end;
    }}


    .rt-source {{
        grid-column: 1 / -1;
    }}


    table.params {{
        width: 100%;
    }}


    table.params th {{
        white-space: normal;
    }}

}}


@media (prefers-reduced-motion: reduce) {{

    * {{
        transition: none !important;

        scroll-behavior: auto !important;
    }}

}}

</style>

</head>


<body>

<div class="wrap">


<header class="top">

    <h1>
        Sonic Logo Stimuli
    </h1>

    <p class="sub">
        Each brand brief is shown once. Its three independent
        generation runs are grouped together so you can compare
        repetition without scrolling through three near-identical
        cards. Select a run to hear the first and best candidates,
        inspect estimator movement, and see exactly which
        synthesiser parameters changed.
    </p>

    <div class="stats">

        <div>
            <b>{len(by_brief)}</b>
            <span>briefs</span>
        </div>

        <div>
            <b>{len(runs)}</b>
            <span>repeated runs</span>
        </div>

        <div>
            <b>{len(runs) * 2}</b>
            <span>audio stimuli</span>
        </div>

        <div>
            <b>{held}/{len(runs)}</b>
            <span>runs stayed in quadrant</span>
        </div>

        <div>
            <b>{tied}/{len(runs)}</b>
            <span>unchanged runs</span>
        </div>

    </div>

</header>


<section class="roles">

    <h2>
        Which estimator is which
    </h2>

    <div class="rgrid">
        {role_cards}
    </div>

    <p class="convention">
        Coach and judge are trained on different corpora and are
        never the same model, so agreement between them carries
        information. Where an estimator was replaced during the
        study, only the one actually used is shown; the filename
        beside each name above is the artefact it came from.
        <b>First candidate</b> is iteration 0 of a run, before any
        revision; <b>best candidate</b> is the best the loop found
        within its iteration cap. Each brief has three independent
        generation runs.
    </p>

</section>


<section class="legend">

    <h2>
        Reading the plots
    </h2>

    <p>
        Valence runs left to right, arousal bottom to top.
        The dashed ring is where the brief aimed. The hollow dot
        is the first candidate, the filled dot is the best
        candidate, and the line between them is how far it travelled.
        {scale_note}
        No point is clipped, and the numbers beneath each plot are
        the coordinates being drawn. The quadrant badge inside a run
        reports {esc(coach_label)}'s verdict because that estimator
        controlled the stopping rule; a held-out judge may place
        the same stimulus differently.
    </p>

    <div class="keys">

        <span class="key">
            <span class="swatch sw-t"></span>
            brief target
        </span>

        <span class="key">
            <span class="swatch sw-n"></span>
            first candidate
        </span>

        <span class="key">
            <span class="swatch sw-o"></span>
            best candidate
        </span>

        <span class="key">
            <b>{esc(coach_label)}</b>
            &nbsp;guided the search
        </span>

        {judge_keys_html}

    </div>

</section>


<div class="filters">

    <span class="lab">
        Quadrant
    </span>

    {quad_btns}

    <span
        class="lab"
        style="margin-left:14px"
    >
        Show briefs containing
    </span>

    <button
        type="button"
        class="chip"
        data-f="held"
        data-v="1"
    >
        held-quadrant run
    </button>

    <button
        type="button"
        class="chip"
        data-f="crossed"
        data-v="1"
    >
        axis-crossing run
    </button>

    <button
        type="button"
        class="chip"
        data-f="tied"
        data-v="1"
    >
        unchanged run
    </button>

    <span
        class="count"
        id="count"
        aria-live="polite"
    ></span>

</div>


{brief_cards}


</div>


<script>

(function () {{

    var root = document;

    var chips =
        Array.prototype.slice.call(
            root.querySelectorAll(
                '.chip'
            )
        );

    var cards =
        Array.prototype.slice.call(
            root.querySelectorAll(
                '.brief-card'
            )
        );

    var count =
        root.getElementById(
            'count'
        );


    /*
     * IMPORTANT:
     *
     * The page NEVER opens in an all-quadrants state.
     *
     * HV / HA is the default active quadrant.
     */
    var active = {{
        quad: {json.dumps(default_quad)},
        held: null,
        crossed: null,
        tied: null
    }};


    function cardMatches(card) {{

        if (
            active.quad
            &&
            card.dataset.quad !== active.quad
        ) {{
            return false;
        }}


        if (
            active.held
            &&
            card.dataset.anyHeld !== active.held
        ) {{
            return false;
        }}


        if (
            active.crossed
            &&
            card.dataset.anyCrossed !== active.crossed
        ) {{
            return false;
        }}


        if (
            active.tied
            &&
            card.dataset.anyTied !== active.tied
        ) {{
            return false;
        }}


        return true;
    }}


    function apply() {{

        var shownBriefs = 0;

        var shownRuns = 0;


        cards.forEach(
            function (card) {{

                var ok =
                    cardMatches(
                        card
                    );


                card.hidden =
                    !ok;


                if (ok) {{

                    shownBriefs += 1;

                    shownRuns += Number(
                        card.dataset.runCount
                        ||
                        0
                    );

                }}

            }}
        );


        count.textContent =
            shownBriefs
            +
            (
                shownBriefs === 1
                ?
                ' brief'
                :
                ' briefs'
            )
            +
            ' · '
            +
            shownRuns
            +
            (
                shownRuns === 1
                ?
                ' run'
                :
                ' runs'
            )
            +
            ' · '
            +
            (shownRuns * 2)
            +
            ' stimuli';

    }}


    function syncChipState() {{

        chips.forEach(
            function (chip) {{

                var f =
                    chip.dataset.f;

                var v =
                    chip.dataset.v;

                var pressed =
                    false;


                if (f === 'quad') {{

                    pressed =
                        active.quad === v;

                }}

                else {{

                    pressed =
                        active[f] === v;

                }}


                chip.setAttribute(
                    'aria-pressed',
                    pressed
                    ?
                    'true'
                    :
                    'false'
                );

            }}
        );

    }}


    /*
     * GLOBAL FILTER BUTTONS
     */
    chips.forEach(
        function (chip) {{

            chip.addEventListener(
                'click',
                function () {{

                    var f =
                        chip.dataset.f;

                    var v =
                        chip.dataset.v;


                    /*
                     * Quadrants are exclusive.
                     *
                     * One quadrant is ALWAYS active.
                     *
                     * Clicking the currently selected quadrant
                     * therefore does NOT switch it off and expose
                     * all 48 repeated-run pairs.
                     */
                    if (f === 'quad') {{

                        active.quad =
                            v;

                    }}

                    else {{

                        active[f] =
                            active[f] === v
                            ?
                            null
                            :
                            v;

                    }}


                    syncChipState();

                    apply();

                }}
            );

        }}
    );


    /*
     * RUN SELECTORS
     *
     * Every brief owns its own Run 1 / Run 2 / Run 3 selector.
     *
     * Switching run changes only that brief's detail section.
     */
    cards.forEach(
        function (card) {{

            var tabs =
                Array.prototype.slice.call(
                    card.querySelectorAll(
                        '[data-run-tab]'
                    )
                );


            var panels =
                Array.prototype.slice.call(
                    card.querySelectorAll(
                        '[data-run-panel]'
                    )
                );


            tabs.forEach(
                function (tab) {{

                    tab.addEventListener(
                        'click',
                        function () {{

                            var run =
                                tab.dataset.runTab;


                            /*
                             * Highlight selected run.
                             */
                            tabs.forEach(
                                function (other) {{

                                    other.setAttribute(
                                        'aria-selected',
                                        other === tab
                                        ?
                                        'true'
                                        :
                                        'false'
                                    );

                                }}
                            );


                            /*
                             * Show corresponding run detail panel.
                             */
                            panels.forEach(
                                function (panel) {{

                                    var show =
                                        panel.dataset.runPanel
                                        ===
                                        run;


                                    panel.hidden =
                                        !show;


                                    /*
                                     * If user changes run while audio
                                     * is playing, stop hidden players.
                                     */
                                    if (!show) {{

                                        Array.prototype
                                            .slice
                                            .call(
                                                panel.querySelectorAll(
                                                    'audio'
                                                )
                                            )
                                            .forEach(
                                                function (audio) {{

                                                    audio.pause();

                                                }}
                                            );

                                    }}

                                }}
                            );

                        }}
                    );

                }}
            );

        }}
    );


    /*
     * ONE AUDIO PLAYER AT A TIME
     *
     * This applies across every brief and every repeated run.
     */
    var players =
        Array.prototype.slice.call(
            root.querySelectorAll(
                'audio'
            )
        );


    players.forEach(
        function (player) {{

            player.addEventListener(
                'play',
                function () {{

                    players.forEach(
                        function (other) {{

                            if (
                                other !== player
                            ) {{

                                other.pause();

                            }}

                        }}
                    );

                }}
            );

        }}
    );


    /*
     * INITIALISE
     *
     * This lights up HV / HA immediately and hides all
     * other quadrants before the user starts browsing.
     */
    syncChipState();

    apply();

}})();

</script>


</body>

</html>
'''

    out = (
        MP3_DIR
        / "index.html"
    )

    MP3_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        html_doc
    )

    names = [
        label
        for _name, label, data in judges
        if data
    ]

    print(
        f"Wrote {out}  "
        f"({len(by_brief)} briefs, "
        f"{len(runs)} repeated runs, "
        f"{len(runs) * 2} stimuli)"
    )

    print(
        f"default quadrant: "
        f"{default_quad or 'none'}"
    )

    print(
        f"coach: "
        f"{coach_label}  "
        f"(models/{coach})"
    )

    for name, label, data in judges:
        if data:
            print(
                f"judge: "
                f"{label}  "
                f"(models/{name})"
            )

    hidden = [
        (
            "estimator_B"
            if f.stem == "h1_estimator_b"
            else f.stem.replace(
                "h1_estimator_b_",
                "",
            )
        )
        for f in sorted(
            ANALYSIS_DIR.glob(
                "h1_estimator_b*.csv"
            )
        )
    ]

    shown = (
        {coach}
        |
        {
            name
            for name, _label, data in judges
            if data
        }
    )

    skipped = [
        h
        for h in hidden
        if h not in shown
    ]

    if skipped and not only:
        print(
            "not plotted "
            "(superseded or the coach re-scored): "
            +
            ", ".join(skipped)
            +
            "   -- use --all to include them"
        )

    if scale == "shared":
        print(
            f"plot axes: shared, "
            f"+/-{max(doms.values()):.2f}"
        )

    else:
        print(
            "plot axes: per estimator -> "
            +
            ", ".join(
                f"{k} +/-{v:.2f}"
                for k, v in doms.items()
            )
        )

    if not names:
        print(
            "note: no h1_estimator_b*.csv found "
            "in data/analysis/ — judge plots omitted."
        )

        print(
            "      Run: "
            "python src/analysis/score_estimator_b.py"
        )


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--scale",
        default="shared",
        choices=[
            "shared",
            "per-judge",
        ],
        help=(
            "'shared' puts every estimator on one axis "
            "so compression is visible; "
            "'per-judge' scales each to its own range "
            "so individual positions are readable"
        ),
    )

    ap.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        help=(
            "also plot superseded estimators "
            "(by default only the latest version "
            "of each is shown)"
        ),
    )

    ap.add_argument(
        "--judges",
        default=None,
        help=(
            "Comma-separated estimator names to plot, in order. "
            "Default: every estimator scored in data/analysis/ "
            "(e.g. estimator_B,estimator_B2)"
        ),
    )

    a = ap.parse_args()

    build(
        (
            [
                j.strip()
                for j in a.judges.split(",")
            ]
            if a.judges
            else None
        ),
        a.scale,
        a.show_all,
    )