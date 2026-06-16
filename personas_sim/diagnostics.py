"""
Sub-group calibration: does a known real-world *gradient* emerge in the
personas, or did a method only hit the overall marginal by luck?

The headline accuracy can be high for the WRONG reason: a mode-collapsed
population (everyone says the same thing) can still average to roughly the
right marginal. The decisive test is whether a known sub-group difference
survives. In UK climate data, younger people are consistently MORE concerned
than older people. If the personas reproduce that age gradient, the
conditioning is doing real work; if the gradient is flat, the right marginal
was a coincidence. (Why a right marginal can still be wrong: models lean toward
'A' and toward uniform once de-biased -- Domínguez-Olmedo et al. 2024; sub-group
fidelity as the real test -- Santurkar et al. 2023. See SOURCES.md.)

Concern score
-------------
Every question here is ordered with option A = most concerned end
("Extremely important" / "Very worried" / "A great deal"). We map a letter to
a concern score = (n_real_options - index), so A scores highest. "Don't know"
style trailing options are included in the ordering but rarely chosen; this is
a directional diagnostic, not a calibrated scale.

A persona's score is its letter's concern score (hard-vote methods) or the
expected concern score under its elicited distribution (the `elicited` method).
"""

YOUNG_MAX_AGE = 40   # personas strictly younger than this = "young"
OLD_MIN_AGE = 60     # personas at least this old = "old"


def _concern_scores(question) -> dict:
    """letter -> concern score, A (most concerned) highest."""
    letters = list(question["options"].keys())
    n = len(letters)
    return {L: n - i for i, L in enumerate(letters)}


def _persona_score(answer, scores) -> float:
    """answer is either a letter (str) or a soft distribution (dict)."""
    if answer is None:
        return None
    if isinstance(answer, dict):
        # expected concern score under the elicited distribution
        return sum(p * scores[L] for L, p in answer.items())
    return scores.get(answer)


def age_gradient(personas, answers, question) -> dict:
    """Compare mean concern score of young vs old personas.

    `personas` and `answers` are aligned lists; each answer is a letter or a
    soft-distribution dict. Returns young/old means, the gradient (young-old),
    and whether its sign matches the expected 'younger more concerned'."""
    scores = _concern_scores(question)
    young, old = [], []
    for p, a in zip(personas, answers):
        s = _persona_score(a, scores)
        if s is None:
            continue
        age = p.get("age")
        if age is None:
            continue
        if age < YOUNG_MAX_AGE:
            young.append(s)
        elif age >= OLD_MIN_AGE:
            old.append(s)

    if not young or not old:
        return {"young_mean": None, "old_mean": None, "gradient": None,
                "as_expected": None, "n_young": len(young), "n_old": len(old)}

    ym, om = sum(young) / len(young), sum(old) / len(old)
    grad = ym - om
    return {"young_mean": ym, "old_mean": om, "gradient": grad,
            "as_expected": grad > 0, "n_young": len(young), "n_old": len(old)}


# --- paired psychographic-axis gradients ------------------------------------
#
# Each axis groups its categories into a HIGH-concern set and a LOW-concern set,
# each with a KNOWN expected direction (high > low) so the gradient is a real
# test, not a guess. Categories outside both sets (e.g. "average openness",
# "no firm party allegiance") are excluded from the split.
#
# Climate-progressive vs climate-skeptic UK parties (not strict left/right):
CONCERNED_PARTIES = {"Labour", "Liberal Democrat", "Green", "SNP / Plaid / other"}
SKEPTIC_PARTIES = {"Conservative", "Reform UK"}
# Big Five openness: higher openness -> more environmental concern.
HIGH_OPENNESS = {"high openness"}
LOW_OPENNESS = {"low openness"}
# Schwartz values: self-transcendence -> more concern; self-enhancement -> less.
CONCERNED_VALUES = {"self-transcendence"}
SKEPTIC_VALUES = {"self-enhancement"}


def axis_gradient(personas, answers, question, attr_key, high_set, low_set) -> dict:
    """Concern gradient between a HIGH-concern group and a LOW-concern group,
    split on `p[attr_key]`. A generalisation of `age_gradient` that works for
    any psychographic axis (politics, openness, values, ...).

    Run it for the axis's own method (the axis IS in the prompt -> expect a clear
    POSITIVE gradient) AND for `demographic` as a control (the axis is NOT in the
    prompt, though the attribute is still stored -> expect ~flat). Positive where
    conditioned, flat where not = proof the axis is actually driving the answers,
    not leaking in via correlated demographics."""
    scores = _concern_scores(question)
    high, low = [], []
    for p, a in zip(personas, answers):
        s = _persona_score(a, scores)
        if s is None:
            continue
        cat = p.get(attr_key)
        if cat in high_set:
            high.append(s)
        elif cat in low_set:
            low.append(s)

    if not high or not low:
        return {"high_mean": None, "low_mean": None, "gradient": None,
                "as_expected": None, "n_high": len(high), "n_low": len(low)}

    hm, lm = sum(high) / len(high), sum(low) / len(low)
    grad = hm - lm
    return {"high_mean": hm, "low_mean": lm, "gradient": grad,
            "as_expected": grad > 0, "n_high": len(high), "n_low": len(low)}


def _tvd(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def persona_dispersion(dists) -> dict:
    """Mean pairwise Total Variation Distance between personas' *elicited* soft
    distributions (the `elicited` method's per-persona output).

    This is the decisive test of whether the personas are actually doing work.
    If the dispersion is ~0, every persona produced essentially the SAME
    distribution -- so a near-perfect averaged marginal is the model *reciting
    an aggregate it already knows*, not *simulating distinct people*. A method
    that recites can't generalise to a question with no published answer, which
    is the whole point of persona simulation. Higher dispersion means the
    personas genuinely individuate.

    `dists` is the list of per-persona distribution dicts (None for failures).
    Returns mean pairwise TVD in [0,1] and the count compared."""
    good = [d for d in dists if d is not None]
    n = len(good)
    if n < 2:
        return {"mean_pairwise_tvd": None, "n": n}
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _tvd(good[i], good[j])
            pairs += 1
    return {"mean_pairwise_tvd": total / pairs, "n": n}
