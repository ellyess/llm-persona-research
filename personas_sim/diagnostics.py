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


# Parties grouped by climate stance (not strict left/right). Climate-progressive
# parties vs climate-skeptic parties in the UK; "no firm party allegiance" is
# excluded from the split. The expected gradient (concerned - skeptic) is
# positive: progressive-party voters are more climate-concerned.
CONCERNED_PARTIES = {"Labour", "Liberal Democrat", "Green", "SNP / Plaid / other"}
SKEPTIC_PARTIES = {"Conservative", "Reform UK"}


def political_gradient(personas, answers, question) -> dict:
    """Concern gradient between climate-progressive-party voters and
    climate-skeptic-party voters. A near-clone of `age_gradient` that splits on
    `p["affiliation"]` instead of age.

    Run it for `psychographic` (politics IS in the prompt -> expect a clear
    POSITIVE gradient) AND for `demographic` as a control (politics is NOT in
    the prompt -> expect ~flat). Positive where conditioned, flat where not =
    proof the political axis is actually driving the answers."""
    scores = _concern_scores(question)
    concerned, skeptic = [], []
    for p, a in zip(personas, answers):
        s = _persona_score(a, scores)
        if s is None:
            continue
        party = p.get("affiliation")
        if party in CONCERNED_PARTIES:
            concerned.append(s)
        elif party in SKEPTIC_PARTIES:
            skeptic.append(s)

    if not concerned or not skeptic:
        return {"green_mean": None, "right_mean": None, "gradient": None,
                "as_expected": None, "n_green": len(concerned),
                "n_right": len(skeptic)}

    gm, rm = sum(concerned) / len(concerned), sum(skeptic) / len(skeptic)
    grad = gm - rm
    return {"green_mean": gm, "right_mean": rm, "gradient": grad,
            "as_expected": grad > 0, "n_green": len(concerned),
            "n_right": len(skeptic)}


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
