"""Model-agnostic LLM client. One interface, swappable backends.

  backend='ollama'    -> local Qwen (primary), OpenAI-style /api/chat
  backend='anthropic' -> Claude Sonnet (optional cross-check), same interface
  backend='mock'      -> deterministic fake responses for dry runs / plumbing tests

Every call is appended to a JSONL log (model, prompts, settings, response, tokens),
which is the per-call audit trail required for RO4.
"""
from __future__ import annotations
import json
import time
import re
import hashlib
import random
import urllib.request
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict = field(default_factory=dict)


def call_seed(*parts) -> int:
    """Reproducible per-call seed from whatever identifies the call.

    Deterministic across runs, distinct between them: seed("B01", 0, 3) is always
    the same number and is never the same as seed("B01", 1, 3).
    """
    key = "|".join(str(x) for x in parts)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class LLMClient:
    def __init__(self, backend="ollama", model="qwen3:8b", temperature=0.7,
                 host="http://localhost:11434", log_path=None, timeout=180, think=False):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.host = host.rstrip("/")
        self.log_path = log_path
        self.timeout = timeout
        self.think = think          # Qwen3 & other reasoning models: False = no hidden chain-of-thought

    def complete(self, system: str, user: str, force_json=True, seed=None) -> LLMResponse:
        """`seed` makes a call reproducible without making it identical to its
        neighbours. Ollama defaults options.seed to 0, so the same prompt returns
        the same completion however high the temperature - which silently turned
        repeated runs and repeated survey trials into exact copies. Passing a seed
        derived from the run and repetition index restores variation and keeps the
        study reproducible."""
        if self.backend == "ollama":
            r = self._ollama(system, user, force_json, seed)
        elif self.backend == "anthropic":
            r = self._anthropic(system, user)
        elif self.backend == "mock":
            r = self._mock(system, user)
        else:
            raise ValueError(f"unknown backend: {self.backend}")
        self._log(system, user, r, seed)
        return r

    # ----- backends -----
    def _ollama(self, system, user, force_json, seed=None):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature,
                        **({"seed": int(seed)} if seed is not None else {})},
        }
        if force_json:
            body["format"] = "json"
        req = urllib.request.Request(
            self.host + "/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            j = json.loads(resp.read())
        return LLMResponse(text=j["message"]["content"], model=self.model, backend="ollama",
                           prompt_tokens=j.get("prompt_eval_count", 0),
                           completion_tokens=j.get("eval_count", 0), raw=j)

    def _anthropic(self, system, user):
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        m = client.messages.create(model=self.model, max_tokens=512, system=system,
                                    messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
        return LLMResponse(text=text, model=self.model, backend="anthropic",
                           prompt_tokens=m.usage.input_tokens,
                           completion_tokens=m.usage.output_tokens)

    def _mock(self, system, user):
        """Dispatch: generator prompts (they carry the parameter schema) get parameter JSON;
        everything else (the audience survey) gets survey JSON."""
        if '"tempo_bpm"' in user:
            return self._mock_params(user)
        return self._mock_survey(system, user)

    def _mock_survey(self, system, user):
        """Fake survey response that reacts to features and persona, for dry-runs only."""
        seed = int(hashlib.md5((system + user).encode()).hexdigest(), 16) % (2 ** 32)
        rnd = random.Random(seed)
        tempo = float((re.search(r"Tempo \(BPM\): ([\d.]+)", user) or [0, "100"])[1])
        centroid = float((re.search(r"centroid \(Hz\): ([\d.]+)", user) or [0, "1800"])[1])
        major = "Mode: major" in user
        extravert = "Outgoing" in system
        stable = "emotionally stable" in system
        arousal = 1 + 8 * ((tempo - 60) / 120 * 0.6 + (centroid - 800) / 2800 * 0.4)
        arousal += 1 if extravert else 0
        valence = 5 + (2 if major else -2) + (1 if stable else -1)
        clip = lambda x, lo, hi: max(lo, min(hi, int(round(x))))
        obj = {"Q1": clip(valence + rnd.uniform(-0.5, 0.5), 1, 9),
               "Q2": clip(arousal + rnd.uniform(-0.5, 0.5), 1, 9)}
        for i in range(3, 13):
            obj[f"Q{i}"] = rnd.randint(1, 5)
        return LLMResponse(text=json.dumps(obj), model="mock", backend="mock")

    def _mock_params(self, user):
        """Fake generator response: valid parameters nudged toward the stated gap."""
        gv = ga = 0.0
        mg = re.search(r"Gap.*?valence ([+-]?\d*\.?\d+), arousal ([+-]?\d*\.?\d+)", user, re.DOTALL)
        if mg:
            gv, ga = float(mg.group(1)), float(mg.group(2))
        cur = lambda pat, d: (re.search(pat, user).group(1) if re.search(pat, user) else d)
        tempo = int(cur(r'"tempo_bpm":\s*(\d+)', "100"))
        center = int(cur(r'"pitch_center_midi":\s*(\d+)', "64"))
        mode = cur(r'"mode":\s*"(major|minor)"', "major")
        tempo = int(min(200, max(40, tempo + ga * 45)))
        center = int(min(84, max(48, center + ga * 6 + gv * 4)))
        if gv > 0.15:
            mode = "major"
        elif gv < -0.15:
            mode = "minor"
        obj = {"tempo_bpm": tempo, "mode": mode, "pitch_center_midi": center,
               "pitch_range": 12, "contour": "rising" if ga >= 0 else "falling",
               "notes_per_beat": 4 if ga > 0.15 else (1 if ga < -0.15 else 2),
               "dynamics": "loud" if ga > 0.15 else ("soft" if ga < -0.15 else "moderate"),
               "articulation": "staccato" if ga > 0 else "legato",
               "instrument": "trumpet" if ga > 0 else "warm pad"}
        return LLMResponse(text=json.dumps(obj), model="mock", backend="mock")

    # ----- logging -----
    def _log(self, system, user, r: LLMResponse, seed=None):
        if not self.log_path:
            return
        rec = {"ts": time.time(), "backend": r.backend, "model": r.model,
               "temperature": self.temperature, "seed": seed,
               "system": system, "user": user,
               "response": r.text, "prompt_tokens": r.prompt_tokens,
               "completion_tokens": r.completion_tokens}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")