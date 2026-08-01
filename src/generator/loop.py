"""LLM parameter proposal/revision + the deterministic optimisation controller.

The controller owns the loop. Each iteration it renders the parameters, scores the
audio with Estimator A, computes the signed valence/arousal gap, and asks the LLM
only to revise the parameters. The LLM never decides whether its own output is good.

  non-optimised = iteration 0 of the run
  optimised     = the best-distance iteration reached within the cap (same run)
"""
from __future__ import annotations
import os
import re
import json
import math
import shutil

from features.extracts import load_audio
from generator.synth import render, validate_params, schema_text
from estimators.data import combined_features


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def _propose(client, prompt: str, retries: int):
    for _ in range(retries + 1):
        r = client.complete("", prompt, force_json=True)
        js = _extract_json(r.text)
        if js:
            try:
                params, err = validate_params(json.loads(js))
                if params is not None:
                    return params
            except json.JSONDecodeError:
                pass
    return None


def propose_initial(client, description, duration, template, retries=3):
    prompt = (template.replace("{duration}", str(duration))
                      .replace("{description}", description)
                      .replace("{schema}", schema_text()))
    return _propose(client, prompt, retries)


def propose_revision(client, params, est, target, template, duration, retries=3):
    gap = (target[0] - est[0], target[1] - est[1])
    prompt = (template.replace("{duration}", str(duration))
                      .replace("{params}", json.dumps(params))
                      .replace("{est_v}", f"{est[0]:+.2f}").replace("{est_a}", f"{est[1]:+.2f}")
                      .replace("{target_v}", f"{target[0]:+.2f}").replace("{target_a}", f"{target[1]:+.2f}")
                      .replace("{gap_v}", f"{gap[0]:+.2f}").replace("{gap_a}", f"{gap[1]:+.2f}")
                      .replace("{schema}", schema_text()))
    return _propose(client, prompt, retries)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class OptimisationController:
    def __init__(self, client, estimator_predict, soundfont, templates,
                 threshold, iteration_cap, duration=3.0, sr=22050, retries=3):
        self.client = client
        self.predict = estimator_predict
        self.sf = soundfont
        self.tpl = templates
        self.threshold = threshold
        self.cap = iteration_cap
        self.duration = duration
        self.sr = sr
        self.retries = retries

    def _score(self, params, wav_path):
        render(params, self.sf, wav_path, self.duration, self.sr)
        y, sr = load_audio(wav_path, sr=self.sr)
        return self.predict(combined_features(y, sr))

    def run(self, brief, run_idx, out_dir):
        target = (brief["target"]["valence"], brief["target"]["arousal"])
        base = f"{brief['id']}_run{run_idx}"
        params = propose_initial(self.client, brief["brand_description"], self.duration,
                                 self.tpl["initial"], self.retries)
        if params is None:
            return None

        history, iter0, best, temps = [], None, None, []
        for it in range(self.cap):
            wav = os.path.join(out_dir, f".{base}_it{it}.wav")
            try:
                est = self._score(params, wav)
            except Exception as e:
                if best is not None:
                    break                          # keep the best so far
                return None                        # first render failed -> re-draw run
            temps.append(wav)
            d = _dist(est, target)
            rec = {"iter": it, "distance": round(d, 4),
                   "est": [round(est[0], 3), round(est[1], 3)], "params": params}
            history.append(rec)
            if it == 0:
                iter0 = {**rec, "wav": wav}
            if best is None or d < best["distance"]:
                best = {**rec, "wav": wav}
            if d <= self.threshold:
                break
            revised = propose_revision(self.client, params, est, target,
                                       self.tpl["revision"], self.duration, self.retries)
            if revised is None:
                break
            params = revised

        nonopt = os.path.join(out_dir, f"{base}_nonopt.wav")
        opt = os.path.join(out_dir, f"{base}_opt.wav")
        shutil.copyfile(iter0["wav"], nonopt)
        shutil.copyfile(best["wav"], opt)
        for t in temps:                            # drop per-iteration temp renders
            if os.path.exists(t):
                os.unlink(t)

        return {
            "brief": brief["id"], "run": run_idx, "target": list(target),
            "non_optimised": {"file": os.path.basename(nonopt), "est": iter0["est"],
                              "distance": iter0["distance"], "params": iter0["params"]},
            "optimised": {"file": os.path.basename(opt), "est": best["est"],
                          "distance": best["distance"], "iteration": best["iter"],
                          "params": best["params"]},
            "iterations": len(history),
            "reached_threshold": best["distance"] <= self.threshold,
            "history": history,
        }