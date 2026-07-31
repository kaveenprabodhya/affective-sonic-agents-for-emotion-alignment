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

    def complete(self, system: str, user: str, force_json=True) -> LLMResponse:
        if self.backend == "ollama":
            r = self._ollama(system, user, force_json)
        elif self.backend == "anthropic":
            r = self._anthropic(system, user)
        elif self.backend == "mock":
            r = self._mock(system, user)
        else:
            raise ValueError(f"unknown backend: {self.backend}")
        self._log(system, user, r)
        return r

    # ----- backends -----
    def _ollama(self, system, user, force_json):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
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
        """Fake response that reacts to features and persona, for dry-runs only."""
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

    # ----- logging -----
    def _log(self, system, user, r: LLMResponse):
        if not self.log_path:
            return
        rec = {"ts": time.time(), "backend": r.backend, "model": r.model,
               "temperature": self.temperature, "system": system, "user": user,
               "response": r.text, "prompt_tokens": r.prompt_tokens,
               "completion_tokens": r.completion_tokens}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")