"""Assemble the audience prompt from config, and parse/validate the Q1-Q12 reply.

Reused by both the pilot and (later) the full audience harness, so the instrument
the agents see is built from questionnaire.yaml + personas.yaml and cannot drift
from Appendix C.
"""
import json
import re

TRAIT_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
KEYS = [f"Q{i}" for i in range(1, 13)]


def render_persona(persona: dict, trait_descriptions: dict) -> str:
    """One pole description per trait as a plain profile (no trait/pole labels, no rating cues)."""
    return "\n".join(f"- {trait_descriptions[t][persona[t]]}" for t in TRAIT_ORDER)


def render_survey(q: dict) -> str:
    lines = [q["instruction"], ""]
    for item in q["dimensional"]:
        a = item["anchors"]
        mid = (item["points"] + 1) // 2
        lines.append(f"{item['id']}. {item['text']}")
        lines.append(f"   ({1} = {a[1]}; {mid} = {a[mid]}; {item['points']} = {a[item['points']]})")
    et = q["emotion_terms"]
    lab = et["labels"]
    lines += ["", "For Q3-Q12, rate how strongly the sonic logo conveys each quality "
              f"({1} = {lab[1]}, {2} = {lab[2]}, {3} = {lab[3]}, {4} = {lab[4]}, {5} = {lab[5]}):"]
    for it in et["items"]:
        lines.append(f"{it['id']}. {et['stem'].replace('{term}', it['term'])}")
    return "\n".join(lines)


SYSTEM_TEMPLATES = {"ocean": "audience_system",
                    "neutral": "audience_system_neutral",
                    "generic": "audience_system_generic"}


def build_messages(persona, feature_block_text, cfg, agent_kind="ocean"):
    system = cfg["prompts"][SYSTEM_TEMPLATES[agent_kind]]
    if agent_kind == "ocean":
        system = system.replace(
            "{persona}", render_persona(persona, cfg["personas"]["trait_descriptions"]))
    user = (cfg["prompts"]["audience_user"]
            .replace("{feature_block}", feature_block_text)
            .replace("{survey}", render_survey(cfg["questionnaire"])))
    return system, user


def parse_validate(text: str):
    """Return (obj, None) if valid per the output contract, else (None, reason)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return None, f"JSON parse error: {e}"
    for k in KEYS:
        if k not in obj:
            return None, f"missing {k}"
        v = obj[k]
        if not isinstance(v, int):
            try:
                v = int(v)
                obj[k] = v
            except (TypeError, ValueError):
                return None, f"{k} not an integer"
        lo, hi = (1, 9) if k in ("Q1", "Q2") else (1, 5)
        if not (lo <= v <= hi):
            return None, f"{k}={v} out of range [{lo},{hi}]"
    return {k: obj[k] for k in KEYS}, None


def run_survey(client, persona, feature_block_text, cfg, retries=3, agent_kind="ocean",
               seed=None):
    """Call the model, validate, retry up to `retries` times. Returns (obj, err, last_response).

    `seed` varies by repetition. Without it the prompt for rep 0, 1 and 2 of a
    persona-stimulus pair is byte-identical, and Ollama's default fixed seed
    returns the same answer three times - so three trials carry one trial's worth
    of information.
    """
    system, user = build_messages(persona, feature_block_text, cfg, agent_kind)
    err = None
    r = None
    for attempt in range(retries + 1):
        r = client.complete(system, user, force_json=True,
                            seed=None if seed is None else seed + attempt)
        obj, err = parse_validate(r.text)
        if obj is not None:
            return obj, None, r
    return None, err, r