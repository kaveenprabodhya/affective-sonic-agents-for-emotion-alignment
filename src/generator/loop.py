"""LLM parameter proposal/revision + the deterministic optimisation controller.

The controller owns the loop. Each iteration it renders the parameters, scores the
audio with Estimator A, computes the signed valence/arousal gap, and asks the LLM
only to revise the parameters. The LLM never decides whether its own output is good.

  non-optimised = iteration 0 of the run
  optimised     = the best iteration reached within the cap (same run)

Stopping rule: distance <= threshold AND the estimate holds the target's sign on
both axes (quadrant condition). Distance alone is a proximity tolerance and does
not protect quadrant membership: when a target sits near the origin relative to
Estimator A's prediction error, a sign-crossed candidate can satisfy the distance
threshold. Set require_quadrant=False to reproduce the distance-only behaviour.
"""
from __future__ import annotations
import os
import re
import json
import math
import shutil
import hashlib
import itertools

from features.extracts import load_audio
from generator.synth import render, validate_params, schema_text, SCHEMA
from estimators.data import combined_features
from llm.client import call_seed


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def _propose(client, prompt: str, retries: int, seed=None):
    # Retry malformed, invalid, or transiently timed-out local model calls.
    # A single Ollama timeout must not terminate the complete generation run.
    for attempt in range(retries + 1):
        try:
            r = client.complete(
                "",
                prompt,
                force_json=True,
                seed=None if seed is None else seed + attempt,
            )
        except TimeoutError:
            continue

        js = _extract_json(r.text)

        if js:
            try:
                params, err = validate_params(
                    json.loads(js)
                )

                if params is not None:
                    return params

            except json.JSONDecodeError:
                pass

    return None


def propose_initial(client, description, duration, template, retries=3, seed=None):
    prompt = (template.replace("{duration}", str(duration))
                      .replace("{description}", description)
                      .replace("{schema}", schema_text()))
    return _propose(client, prompt, retries, seed)


def propose_revision(
    client,
    description,
    params,
    est,
    target,
    template,
    duration,
    retries=3,
    seed=None,
    history=None,
):
    gap = (
        target[0] - est[0],
        target[1] - est[1],
    )

    if history:
        history_text = "\n".join(
            (
                f"- iteration {h['iter']}: "
                f"distance={h['distance']:.4f}, "
                f"quadrant_ok={h['quadrant_ok']}, "
                f"estimated_va={h['est']}, "
                f"parameters={json.dumps(h['params'], sort_keys=True)}"
            )
            for h in history[-8:]
        )
    else:
        history_text = "No previous candidates have been evaluated."

    prompt = (
        template
        .replace("{duration}", str(duration))
        .replace("{description}", description)
        .replace("{params}", json.dumps(params))
        .replace("{est_v}", f"{est[0]:+.2f}")
        .replace("{est_a}", f"{est[1]:+.2f}")
        .replace("{target_v}", f"{target[0]:+.2f}")
        .replace("{target_a}", f"{target[1]:+.2f}")
        .replace("{gap_v}", f"{gap[0]:+.2f}")
        .replace("{gap_a}", f"{gap[1]:+.2f}")
        .replace("{history}", history_text)
        .replace("{schema}", schema_text())
    )

    return _propose(
        client,
        prompt,
        retries,
        seed,
    )
    

# Local deterministic optimisation steps for the integer-valued
# synthesis parameters. All categorical alternatives come directly
# from the synthesis schema.
_LOCAL_INT_STEPS = {
    "tempo_bpm": 20,
    "pitch_center_midi": 4,
    "pitch_range": 4,
}


def _param_key(params):
    """Stable representation used to detect previously evaluated candidates."""
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
    )


def _local_alternatives(params):
    """Return legal local alternatives for every synthesis parameter."""
    out = {}

    for name, spec in SCHEMA.items():
        current = params[name]

        if spec[0] == "int":
            lo = int(spec[1])
            hi = int(spec[2])
            step = _LOCAL_INT_STEPS[name]

            values = []

            for value in (
                current - step,
                current + step,
            ):
                value = max(
                    lo,
                    min(
                        hi,
                        int(value),
                    ),
                )

                if (
                    value != current
                    and value not in values
                ):
                    values.append(value)

            out[name] = values

        else:
            out[name] = [
                value
                for value in spec[1]
                if value != current
            ]

    return out


def _single_neighbours(params):
    """Candidates differing from the current best on one parameter."""
    alts = _local_alternatives(
        params
    )

    candidates = []

    for name, values in alts.items():
        for value in values:
            cand = dict(params)
            cand[name] = value
            candidates.append(cand)

    return candidates


def _pair_neighbours(params):
    """Candidates differing on two parameters.

    Pair moves are considered only after unseen one-parameter moves around
    the current best have been exhausted.
    """
    alts = _local_alternatives(
        params
    )

    active = [
        name
        for name, values in alts.items()
        if values
    ]

    candidates = []

    for name1, name2 in itertools.combinations(
        active,
        2,
    ):
        for value1 in alts[name1]:
            for value2 in alts[name2]:
                cand = dict(params)
                cand[name1] = value1
                cand[name2] = value2
                candidates.append(cand)

    return candidates


def _deterministic_candidate_order(
    candidates,
    brief_id,
    run_idx,
    best_params,
    layer,
):
    """Reproducible but brief/run-specific ordering of local candidates."""
    salt = (
        f"{brief_id}|{run_idx}|{layer}|"
        f"{_param_key(best_params)}"
    )

    def rank(candidate):
        text = (
            salt
            + "|"
            + _param_key(candidate)
        )

        return hashlib.sha256(
            text.encode()
        ).hexdigest()

    return sorted(
        candidates,
        key=rank,
    )

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _quadrant_ok(est, target):
    """True if the estimate sits on the same side of both axes as the target.

    An axis whose target is exactly 0 carries no sign constraint. An estimate of
    exactly 0 on a constrained axis is treated as NOT satisfying the condition:
    it sits on the boundary, so quadrant membership is undefined.
    """
    for e, t in zip(est, target):
        if t == 0:
            continue
        if e * t <= 0:
            return False
    return True


def _better(cand, cur, require_quadrant):
    """Candidate-selection rule. With the quadrant condition on, a candidate that
    holds the intended quadrant always beats one that does not, regardless of
    distance; ties within the same class are broken on distance.
    """
    if require_quadrant and cand["quadrant_ok"] != cur["quadrant_ok"]:
        return cand["quadrant_ok"]
    return cand["distance"] < cur["distance"]


class OptimisationController:
    def __init__(self, client, estimator_predict, soundfont, templates,
                 threshold, iteration_cap, duration=3.0, sr=22050, retries=3,
                 require_quadrant=True):
        self.client = client
        self.predict = estimator_predict
        self.sf = soundfont
        self.tpl = templates
        self.threshold = threshold
        self.cap = iteration_cap
        self.duration = duration
        self.sr = sr
        self.retries = retries
        self.require_quadrant = require_quadrant

    def _score(self, params, wav_path):
        render(params, self.sf, wav_path, self.duration, self.sr)
        y, sr = load_audio(wav_path, sr=self.sr)
        return self.predict(combined_features(y, sr))

    def run(self, brief, run_idx, out_dir):
        target = (brief["target"]["valence"], brief["target"]["arousal"])
        base = f"{brief['id']}_run{run_idx}"
        # Seeded per brief and run, so the three runs of a brief are genuinely
        # three draws rather than three copies, while the study stays reproducible.
        params = propose_initial(self.client, brief["brand_description"], self.duration,
                                 self.tpl["initial"], self.retries,
                                 seed=call_seed(brief["id"], run_idx, "init"))
        if params is None:
            return None

        history, iter0, best, temps = [], None, None, []
        seen_param_keys = set()

        # Iteration 0 is the brand-conditioned parameterisation proposed
        # by the language model. Subsequent candidates are generated by
        # deterministic local search around the best candidate found so far.
        while len(history) < self.cap:

            it = len(history)

            wav = os.path.join(
                out_dir,
                f".{base}_it{it}.wav",
            )

            try:
                est = self._score(
                    params,
                    wav,
                )

            except Exception:
                if best is not None:
                    break
                return None

            temps.append(
                wav
            )

            d = _dist(
                est,
                target,
            )

            quad_ok = _quadrant_ok(
                est,
                target,
            )

            rec = {
                "iter": it,
                "distance": round(d, 4),
                "quadrant_ok": quad_ok,
                "est": [
                    round(est[0], 3),
                    round(est[1], 3),
                ],
                "params": dict(params),
            }

            history.append(
                rec
            )

            seen_param_keys.add(
                _param_key(params)
            )

            if it == 0:
                iter0 = {
                    **rec,
                    "wav": wav,
                }

            if (
                best is None
                or _better(
                    rec,
                    best,
                    self.require_quadrant,
                )
            ):
                best = {
                    **rec,
                    "wav": wav,
                }

            # Predefined stopping rule:
            # proximity threshold plus intended-quadrant membership.
            if (
                best["distance"] <= self.threshold
                and (
                    best["quadrant_ok"]
                    or not self.require_quadrant
                )
            ):
                break

            # --------------------------------------------------------
            # Deterministic local search around the best candidate.
            #
            # First consider unseen one-parameter changes. If all such
            # neighbours have already been evaluated, permit two-
            # parameter moves so the search can escape a coordinate-
            # wise local optimum.
            # --------------------------------------------------------

            candidates = [
                candidate
                for candidate in _single_neighbours(
                    best["params"]
                )
                if _param_key(candidate)
                not in seen_param_keys
            ]

            layer = "single"

            if not candidates:
                candidates = [
                    candidate
                    for candidate in _pair_neighbours(
                        best["params"]
                    )
                    if _param_key(candidate)
                    not in seen_param_keys
                ]

                layer = "pair"

            if not candidates:
                break

            candidates = _deterministic_candidate_order(
                candidates,
                brief["id"],
                run_idx,
                best["params"],
                layer,
            )

            # One new candidate is rendered and scored per iteration.
            # Recomputing the neighbourhood after every evaluation means
            # that any improvement immediately becomes the new search centre.
            params = candidates[0]

        nonopt = os.path.join(out_dir, f"{base}_nonopt.wav")
        opt = os.path.join(out_dir, f"{base}_opt.wav")
        shutil.copyfile(iter0["wav"], nonopt)
        shutil.copyfile(best["wav"], opt)
        for t in temps:                            # drop per-iteration temp renders
            if os.path.exists(t):
                os.unlink(t)

        return {
            "brief": brief["id"], "run": run_idx, "target": list(target),
            "require_quadrant": self.require_quadrant,
            "non_optimised": {"file": os.path.basename(nonopt), "est": iter0["est"],
                              "distance": iter0["distance"],
                              "quadrant_ok": iter0["quadrant_ok"], "params": iter0["params"]},
            "optimised": {"file": os.path.basename(opt), "est": best["est"],
                          "distance": best["distance"], "iteration": best["iter"],
                          "quadrant_ok": best["quadrant_ok"], "params": best["params"]},
            "iterations": len(history),
            "reached_distance": best["distance"] <= self.threshold,
            "quadrant_held": best["quadrant_ok"],
            "reached_threshold": (best["distance"] <= self.threshold
                                  and (best["quadrant_ok"] or not self.require_quadrant)),
            "history": history,
        }