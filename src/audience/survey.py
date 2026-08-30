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
    """Render the complete Q1-Q12 survey shown to every audience agent."""

    lines = [
        q["instruction"],
        "",
    ]

    # Q1-Q2: show every point on the 1-9 scale explicitly.
    for item in q["dimensional"]:

        anchors = item["anchors"]

        lines.append(
            f"{item['id']}. {item['text']}"
        )

        scale_text = "; ".join(
            f"{value} = {anchors[value]}"
            for value in range(
                1,
                item["points"] + 1
            )
        )

        lines.append(
            f"   ({scale_text})"
        )

    # Q3-Q12
    et = q["emotion_terms"]
    labels = et["labels"]

    lines += [
        "",
        "For Q3-Q12, rate how strongly the sonic logo "
        "conveys each quality "
        f"(1 = {labels[1]}; "
        f"2 = {labels[2]}; "
        f"3 = {labels[3]}; "
        f"4 = {labels[4]}; "
        f"5 = {labels[5]}):",
    ]

    for item in et["items"]:

        lines.append(
            f"{item['id']}. "
            f"{et['stem'].replace('{term}', item['term'])}"
        )

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


def run_survey(
    client,
    persona,
    feature_block_text,
    cfg,
    retries=3,
    agent_kind="ocean",
    seed=None,
):
    """Call the model, validate, and repair invalid responses.

    The first call uses the normal audience protocol.

    If validation fails, the next call receives the same survey and acoustic
    evidence plus the exact mechanical validation error. The repair instruction
    changes only response-format compliance; it does not provide targets,
    intended emotion, condition labels, estimator outputs, or desired answers.
    """

    system, user = build_messages(
        persona,
        feature_block_text,
        cfg,
        agent_kind,
    )

    err = None
    r = None

    # First attempt uses the untouched experimental prompt.
    current_user = user

    for attempt in range(retries + 1):

        r = client.complete(
            system,
            current_user,
            force_json=True,
            seed=(
                None
                if seed is None
                else seed + attempt
            ),
        )

        obj, err = parse_validate(
            r.text
        )

        if obj is not None:
            return obj, None, r

        # The next attempt is a mechanical validation repair.
        # No emotional direction or desired score is supplied.
        current_user = (
            user
            + "\n\n"
            + "VALIDATION CORRECTION\n"
            + "Your previous JSON response was invalid for the following "
              "mechanical reason:\n"
            + f"{err}\n\n"
            + "Previous response:\n"
            + f"{r.text}\n\n"
            + "Return the complete Q1-Q12 JSON object again.\n"
            + "Preserve answers that already satisfy their permitted scale.\n"
            + "For any invalid item, reconsider that item using the same "
              "acoustic evidence and select a value inside its permitted "
              "scale.\n\n"
            + "Mandatory ranges:\n"
            + "- Q1 and Q2: integers 1 through 9.\n"
            + "- Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11 and Q12: "
              "integers 1 through 5 only.\n"
            + "- Values 6, 7, 8 and 9 are invalid for Q3-Q12.\n\n"
            + "Return JSON only."
        )

    return None, err, r