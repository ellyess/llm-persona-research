"""
Build personas. Multiple methods so the experiment has something to COMPARE:

  method="demographic" : sample attributes to match UK 16+ proportions and
                         write a short factual persona.
  method="rich"        : same sampled attributes, fuller first-person backstory.
  method="network"     : same persona text as `rich`; the distinctive behaviour
                         lives in the two-pass discussion runner in run.py.

A baseline (no persona at all) lives in run.py, not here.
"""

import random

from .config import DEMOGRAPHICS, POLITICS

# Which demographic attributes are sampled and written into persona prompts.
# This is the "silicon sampling" approach -- condition the model on demographics
# to recover a subpopulation's opinion distribution (Argyle et al. 2023 -- see
# SOURCES.md).
# `ethnicity` is present in DEMOGRAPHICS (faithful to the source) but left out
# here: it adds noise without a strong link to climate opinion. Keeping this
# list explicit makes the modelling choice reviewable and easy to extend.
PROMPT_ATTRIBUTES = ["generation", "sex", "region", "education", "income"]

# Buckets that can't anchor a coherent persona ("you are someone who prefers
# not to say their income"). Dropped, then the remaining categories are
# renormalised, so e.g. income is sampled over the 6 real bands only.
UNINFORMATIVE = {"prefer not to say", "dont know", "don't know"}

# Yale report `generation`, not age bands. Map each generation to a plausible
# 2024 age range (respecting the 16+ frame) so the prompt carries a concrete
# age and the age-gradient stretch-check still works.
GEN_AGE_RANGES = {
    "Gen Z (1997 or later)":    (16, 27),
    "Millennials (1981-1996)":  (28, 43),
    "Gen X (1965-1980)":        (44, 59),
    "Baby Boomers (1946-1964)": (60, 78),
    "Silent (1928-1945)":       (79, 95),
}

# Natural-language phrasings so prompts read like a person, not a data row.
_SEX_PHRASE = {
    "male": "man",
    "female": "woman",
    "another gender identity": "person of another gender identity",
}
_EDU_PHRASE = {
    "qualifications level 1 or below":     "below GCSE level",
    "GCSE / O-level / NVQ2":               "GCSE / O-level",
    "A-level / Scottish Higher / NVQ3":    "A-level / Scottish Higher",
    "other higher education below degree": "higher education below degree level",
    "degree level or above":              "degree level or above",
    "another qualification":              "another type of qualification",
    "no qualification":                   "no formal qualifications",
}
_INCOME_PHRASE = {
    "<26K":      "under £26,000",
    "26K-52K":   "between £26,000 and £52,000",
    "52K-75K":   "between £52,000 and £75,000",
    "75K-100K":  "between £75,000 and £100,000",
    "100K-150K": "between £100,000 and £150,000",
    "150K+":     "over £150,000",
}
# Light political-identity sentence for the `psychographic` method. Understated
# on purpose -- the goal is a plausible lean, not a caricature.
_POLITICS_PHRASE = {
    "Labour":                   "Politically you lean towards Labour.",
    "Conservative":             "Politically you lean towards the Conservatives.",
    "Reform UK":                "Politically you lean towards Reform UK.",
    "Liberal Democrat":         "Politically you lean towards the Liberal Democrats.",
    "Green":                    "Politically you lean towards the Green Party.",
    "SNP / Plaid / other":      "Politically you support a smaller or nationalist party.",
    "no firm party allegiance": "You don't have a firm political allegiance.",
}


def _clean_dist(dist: dict) -> dict:
    """Drop uninformative buckets and renormalise the rest to sum to 1."""
    kept = {k: v for k, v in dist.items() if k.lower() not in UNINFORMATIVE}
    total = sum(kept.values())
    return {k: v / total for k, v in kept.items()}


def _sample_attributes(rng: random.Random) -> dict:
    """Draw one persona's attributes from the (cleaned) source distributions."""
    attrs = {}
    for field in PROMPT_ATTRIBUTES:
        dist = _clean_dist(DEMOGRAPHICS[field])
        cats, probs = list(dist.keys()), list(dist.values())
        attrs[field] = rng.choices(cats, weights=probs)[0]
    return attrs


def _age_from_generation(generation: str, rng: random.Random) -> int:
    lo, hi = GEN_AGE_RANGES[generation]
    return rng.randint(lo, hi)


def _sample_affiliation(rng: random.Random) -> str:
    dist = _clean_dist(POLITICS)
    cats, probs = list(dist.keys()), list(dist.values())
    return rng.choices(cats, weights=probs)[0]


def build_personas(n: int, method: str = "demographic", seed: int = 0):
    """Return a list of n dicts: {attributes, age, affiliation, system_prompt}.

    Political affiliation is drawn from a SEPARATE RNG stream (seed + 1) so the
    demographic draws -- and therefore every existing method's personas and
    numbers -- stay byte-identical to before this axis existed. Affiliation is
    stored on every persona for the political-gradient diagnostic, but only the
    `psychographic` prompt actually mentions it."""
    rng = random.Random(seed)
    pol_rng = random.Random(seed + 1)
    personas = []
    for _ in range(n):
        attrs = _sample_attributes(rng)
        age = _age_from_generation(attrs["generation"], rng)
        affiliation = _sample_affiliation(pol_rng)
        attrs["affiliation"] = affiliation
        personas.append({
            "attributes": attrs,
            "age": age,
            "affiliation": affiliation,
            "system_prompt": _make_prompt(attrs, age, method),
        })
    return personas


def _describe(attrs: dict, age: int) -> str:
    """Build the natural-language clause shared by the persona prompts."""
    sex = _SEX_PHRASE.get(attrs["sex"], attrs["sex"])
    region = attrs["region"]
    edu = _EDU_PHRASE.get(attrs["education"], attrs["education"])
    income = _INCOME_PHRASE.get(attrs["income"], attrs["income"])
    return (
        f"a {age}-year-old {sex} living in {region}. "
        f"Your education level is {edu} and your household income is {income}"
    )


# Anti-sycophancy framing. LLMs default to the "responsible"/central answer,
# which collapses a population onto one option (mode collapse) and shifts the
# distribution. Explicitly licensing disagreement is a cheap, first-principles
# fix for that failure mode. Appended to every persona prompt.
# (The RLHF agreeable/cautious default this counters: Sharma et al. 2023 -- see
# SOURCES.md.)
_ANTISYC = (
    " There are no right or wrong answers, and many real people genuinely "
    "disagree on this. Answer as THIS specific person honestly would, even if "
    "it isn't the cautious or socially-approved view -- do not hedge."
)


def _make_prompt(attrs: dict, age: int, method: str) -> str:
    desc = _describe(attrs, age)

    if method == "demographic":
        return (
            f"You are {desc}. "
            f"Answer the survey question as this person would." + _ANTISYC
        )

    if method == "psychographic":
        # = `demographic` + ONE political-identity sentence. Politics is the
        # strongest UK climate-attitude driver absent from the Yale demographics
        # (psychographics beyond demographics; see SOURCES.md). The political
        # clause is the only thing that differs from `demographic`, so any
        # change in the distribution is attributable to that axis.
        pol = _POLITICS_PHRASE.get(attrs.get("affiliation"), "")
        return (
            f"You are {desc}. {pol} "
            f"Answer the survey question as this person would." + _ANTISYC
        )

    if method == "rich":
        return (
            f"Adopt the persona of {desc}. "
            f"Think about this person's daily life, livelihood, community, and "
            f"what they tend to worry about, then answer the survey question "
            f"honestly as they would -- not as you think is correct." + _ANTISYC
        )

    if method == "network":
        # Same persona text as `rich`. The method's distinctive behaviour is
        # in the multi-agent discussion runner in run.py.
        return _make_prompt(attrs, age, method="rich")

    if method == "elicited":
        # Same persona text as `rich`; the distinctive behaviour is the
        # distribution-elicitation question format (see elicitation_prompt).
        return _make_prompt(attrs, age, method="rich")

    raise ValueError(f"unknown method: {method!r}")


# ---- question / discussion prompts ------------------------------------------

def question_block(question: dict, question_text: str = None) -> str:
    """The standard user-message asking the persona to pick a letter.
    `question_text` lets the consistency check substitute a rephrasing while
    keeping the option labels identical."""
    opts = "\n".join(f"{k}) {v}" for k, v in question["options"].items())
    text = question_text or question["text"]
    return f"{text}\n{opts}\nReply with only the letter."


def elicitation_prompt(question: dict) -> str:
    """`elicited` method (verbalized sampling): instead of forcing ONE letter,
    ask the persona for their probability over each option. Averaging these
    soft per-persona distributions recovers within-person uncertainty and is
    far less susceptible to mode collapse than averaging hard votes.

    Verbalized sampling as a mode-collapse fix: Zhang et al. 2025. Self-reported
    (vs. logit-derived) probabilities, and their calibration caveat: Lin et al.
    2022. See SOURCES.md."""
    opts = "\n".join(f"{k}) {v}" for k, v in question["options"].items())
    letters = ", ".join(question["options"].keys())
    return (
        f"{question['text']}\n{opts}\n\n"
        f"You are uncertain, like a real person. Estimate the probability (as a "
        f"percentage) that you would pick each option on different days or in "
        f"different moods. Give a number for every option ({letters}); they "
        f"should sum to about 100.\n"
        f"Reply ONLY in the form 'A: <n>, B: <n>, ...' with no other text."
    )


def opinion_prompt(question: dict) -> str:
    """Round-1 prompt for the `network` method: ask the persona for a brief
    free-text opinion on the topic -- NOT for the multiple-choice letter.
    The point is to surface substantive views the group will then react to.
    LLM agents forming/propagating opinions through conversation: Park et al.
    2023 (Generative Agents) -- see SOURCES.md."""
    return (
        f"You are about to discuss this with a small group before answering:\n\n"
        f"  \"{question['text']}\"\n\n"
        f"Share your honest view in ONE short sentence (max ~25 words), "
        f"in your persona's voice. Do not pick an option yet -- just say "
        f"what you think about the topic."
    )


def question_block_after_discussion(question: dict, peer_opinions) -> str:
    """Round-2 prompt for the `network` method: show the persona what their
    discussion-group peers said in round 1, then ask them to pick a letter.
    They may stick or shift after hearing the others -- this is what models
    social-influence dynamics in toy form."""
    peers = "\n".join(f"  - \"{op.strip()}\"" for op in peer_opinions if op and op.strip())
    intro = (
        "In your small discussion group, the other members said:\n"
        f"{peers}\n\n"
        "Now, after hearing them out, give YOUR own honest answer. "
        "You may stick with your initial view or change your mind.\n\n"
    )
    return intro + question_block(question)


def make_groups(n: int, group_size: int, seed: int = 0):
    """Partition indices [0..n) into groups of size `group_size` (last group
    may be smaller). Deterministic shuffle so groups vary between seeds."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i:i + group_size] for i in range(0, n, group_size)]
