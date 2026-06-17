"""
Unit tests for the metric + parsing logic -- the parts where a silent bug would
quietly corrupt every reported number. Run offline, no API needed:

    python -m pytest tests/ -q          # or: python tests/test_evaluate.py
"""

import os
import sys

# Make the repo root importable whether run via pytest or as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personas_sim.evaluate import (
    to_distribution, unparseable_rate, total_variation, distribution_accuracy,
    jensen_shannon, entropy, peak, evaluate, evaluate_distribution,
)
from personas_sim.llm import parse_letter, parse_distribution
from personas_sim.diagnostics import (
    age_gradient, persona_dispersion, axis_gradient,
    CONCERNED_PARTIES, SKEPTIC_PARTIES, HIGH_OPENNESS, LOW_OPENNESS,
    CONCERNED_VALUES, SKEPTIC_VALUES,
)
from personas_sim.personas import build_personas

# A 4-option question stub (no LLM involved).
Q = {
    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
    "ground_truth": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
}


def approx(x, y, tol=1e-9):
    return abs(x - y) <= tol


# --- distribution + metrics --------------------------------------------------

def test_to_distribution_excludes_none_and_normalises():
    d = to_distribution(["A", "A", "B", None], Q)
    assert approx(d["A"], 2 / 3) and approx(d["B"], 1 / 3)
    assert approx(sum(d.values()), 1.0)


def test_unparseable_rate():
    assert approx(unparseable_rate(["A", None, None, "B"]), 0.5)
    assert approx(unparseable_rate([]), 0.0)


def test_tvd_identity_and_disjoint():
    p = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    q = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
    assert approx(total_variation(p, p), 0.0)
    assert approx(total_variation(p, q), 1.0)
    assert approx(distribution_accuracy(p, q), 0.0)


def test_jsd_zero_when_equal_and_bounded():
    p = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    assert approx(jensen_shannon(p, p), 0.0)
    q = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    r = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
    assert 0.0 <= jensen_shannon(q, r) <= 1.0 + 1e-9


def test_peak_and_entropy_flag_mode_collapse():
    collapsed = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    uniform = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    assert approx(peak(collapsed), 1.0) and approx(peak(uniform), 0.25)
    assert approx(entropy(collapsed), 0.0)
    assert approx(entropy(uniform), 2.0)        # log2(4)


def test_evaluate_block_shape():
    r = evaluate(["A", "B", "C", "D"], Q)
    assert approx(r["distribution_accuracy"], 1.0)   # uniform == ground truth
    assert r["n"] == 4 and approx(r["unparseable_rate"], 0.0)
    assert "peak" in r and "entropy" in r


def test_evaluate_distribution_path():
    r = evaluate_distribution({"A": .25, "B": .25, "C": .25, "D": .25}, Q,
                              unparseable=0.1, n=10)
    assert approx(r["distribution_accuracy"], 1.0)
    assert approx(r["unparseable_rate"], 0.1) and r["n"] == 10


# --- parsing -----------------------------------------------------------------

def test_parse_letter_standalone_only():
    valid = ["A", "B", "C", "D"]
    assert parse_letter("I'd say C.", valid) == "C"
    assert parse_letter("Answer: b", valid) == "B"      # case-insensitive
    assert parse_letter("no option here", valid) is None
    # 'A' embedded in a word must not match; first standalone letter wins
    assert parse_letter("Anyway, D", valid) == "D"


def test_parse_distribution_formats_and_normalise():
    valid = ["A", "B", "C", "D"]
    d = parse_distribution("A: 20, B: 30, C: 25, D: 25", valid)
    assert approx(sum(d.values()), 1.0) and approx(d["B"], 0.30)
    d2 = parse_distribution("A) 50%  B) 50%", valid)
    assert approx(d2["A"], 0.5) and approx(d2["C"], 0.0)
    assert parse_distribution("mostly B I think", valid) is None


# --- diagnostics -------------------------------------------------------------

def _persona(age):
    return {"age": age, "attributes": {}, "system_prompt": ""}


def test_age_gradient_sign():
    importance = {
        "options": {"A": "x", "B": "y", "C": "z"},
        "ground_truth": {"A": 0.34, "B": 0.33, "C": 0.33},
    }
    personas = [_persona(20), _persona(25), _persona(70), _persona(75)]
    answers = ["A", "A", "C", "C"]   # young most concerned, old least
    g = age_gradient(personas, answers, importance)
    assert g["gradient"] > 0 and g["as_expected"] is True


def test_persona_dispersion_zero_when_identical():
    same = {"A": 0.5, "B": 0.5, "C": 0.0}
    out = persona_dispersion([dict(same), dict(same), dict(same)])
    assert approx(out["mean_pairwise_tvd"], 0.0) and out["n"] == 3
    assert persona_dispersion([same])["mean_pairwise_tvd"] is None  # n<2


# --- psychographic axes ------------------------------------------------------

def test_personas_carry_all_axis_attributes():
    ps = build_personas(10, method="psychographic", seed=0)
    assert all({"affiliation", "openness", "values"} <= set(p) for p in ps)


def test_each_axis_method_injects_only_its_clause():
    # Each axis method adds ONE distinctive clause; the others stay absent.
    poli = build_personas(8, "psychographic", 0)
    opn = build_personas(8, "openness", 0)
    val = build_personas(8, "values", 0)
    demo = build_personas(8, "demographic", 0)
    assert all("Politically" in p["system_prompt"] or "allegiance" in p["system_prompt"]
               for p in poli)
    assert all(("open" in p["system_prompt"].lower() or "curious" in p["system_prompt"]
                or "conventional" in p["system_prompt"]) for p in opn)
    assert all("What matters most" in p["system_prompt"] for p in val)
    # demographic carries none of the axis clauses
    assert all("Politically" not in p["system_prompt"]
               and "What matters most" not in p["system_prompt"] for p in demo)


def test_education_age_plausibility_constraint():
    # No persona should hold a qualification it is too young to have reached,
    # and the education marginal must be preserved (we only adjust the free age).
    ps = build_personas(300, "demographic", seed=0)
    floors = {"A-level / Scottish Higher / NVQ3": 18,
              "other higher education below degree": 19,
              "degree level or above": 21}
    for p in ps:
        edu = p["attributes"]["education"]
        if edu in floors:
            assert p["age"] >= floors[edu], (edu, p["age"])
    # education categories still appear (marginal not collapsed by the constraint)
    seen = {p["attributes"]["education"] for p in ps}
    assert "degree level or above" in seen and len(seen) >= 4


def test_composite_prompt_carries_all_three_clauses():
    ps = build_personas(8, "composite", 0)
    for p in ps:
        s = p["system_prompt"]
        assert "Politically" in s or "allegiance" in s          # political
        assert ("curious" in s or "conventional" in s or "open" in s.lower())  # openness
        assert "What matters most" in s                          # values


def test_separate_rng_leaves_demographic_personas_unchanged():
    # Axes are drawn from seed+1/2/3, so demographic prompts must be identical
    # across rebuilds (deterministic and independent of the axis draws).
    a = [p["system_prompt"] for p in build_personas(10, "demographic", 0)]
    b = [p["system_prompt"] for p in build_personas(10, "demographic", 0)]
    assert a == b
    assert all("Politically" not in s for s in a)


_IMPORTANCE = {
    "options": {"A": "x", "B": "y", "C": "z"},
    "ground_truth": {"A": 0.34, "B": 0.33, "C": 0.33},
}


def test_axis_gradient_political_sign():
    personas = [
        {"affiliation": "Green"}, {"affiliation": "Labour"},
        {"affiliation": "Conservative"}, {"affiliation": "Reform UK"},
    ]
    answers = ["A", "A", "C", "C"]   # progressive concerned, skeptic not
    g = axis_gradient(personas, answers, _IMPORTANCE, "affiliation",
                      CONCERNED_PARTIES, SKEPTIC_PARTIES)
    assert g["gradient"] > 0 and g["as_expected"] is True
    assert g["n_high"] == 2 and g["n_low"] == 2


def test_axis_gradient_exploratory_makes_no_claim():
    # expected="exploratory" reports the gradient but as_expected stays None
    g = axis_gradient(
        [{"openness": "high openness"}, {"openness": "low openness"}],
        ["A", "C"], _IMPORTANCE, "openness", HIGH_OPENNESS, LOW_OPENNESS,
        expected="exploratory")
    assert g["gradient"] is not None and g["as_expected"] is None


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    from personas_sim.evaluate import bootstrap_accuracy_ci
    # An interior point estimate (~0.85), not the boundary, so a percentile CI
    # can bracket it from both sides.
    answers = ["A"] * 40 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20
    point = evaluate(answers, Q)["distribution_accuracy"]
    lo, hi = bootstrap_accuracy_ci(answers, Q, n_boot=1000, seed=0)
    assert lo is not None and 0.0 <= lo <= hi <= 1.0
    assert lo - 1e-9 <= point <= hi + 1e-9          # CI brackets the estimate
    assert bootstrap_accuracy_ci(answers, Q, n_boot=1000, seed=0) == (lo, hi)  # deterministic
    assert bootstrap_accuracy_ci([], Q) == (None, None)               # empty


def test_bootstrap_ci_handles_soft_distributions():
    from personas_sim.evaluate import bootstrap_accuracy_ci
    dists = [{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}] * 20
    lo, hi = bootstrap_accuracy_ci(dists, Q, n_boot=300, seed=1)
    assert lo is not None and lo <= hi


def test_ai_question_present_and_evaluates():
    from personas_sim.config import QUESTIONS
    ai = [q for q in QUESTIONS if q.get("topic") == "ai"]
    assert ai, "expected an AI-topic question"
    q = ai[0]
    r = evaluate(["A", "B", "C", "D", "E"], q)
    assert 0.0 <= r["distribution_accuracy"] <= 1.0 and r["n"] == 5
    assert abs(sum(q["ground_truth"].values()) - 1.0) < 1e-9


def test_axis_gradient_openness_and_values():
    # openness: high more concerned than low
    op = axis_gradient(
        [{"openness": "high openness"}, {"openness": "low openness"}],
        ["A", "C"], _IMPORTANCE, "openness", HIGH_OPENNESS, LOW_OPENNESS)
    assert op["gradient"] > 0 and op["as_expected"] is True
    # values: self-transcendence more concerned than self-enhancement
    va = axis_gradient(
        [{"values": "self-transcendence"}, {"values": "self-enhancement"}],
        ["A", "C"], _IMPORTANCE, "values", CONCERNED_VALUES, SKEPTIC_VALUES)
    assert va["gradient"] > 0 and va["as_expected"] is True
    # a category outside both sets is excluded from the split
    none = axis_gradient(
        [{"values": "conservation"}, {"values": "openness to change"}],
        ["A", "C"], _IMPORTANCE, "values", CONCERNED_VALUES, SKEPTIC_VALUES)
    assert none["gradient"] is None and none["n_high"] == 0 and none["n_low"] == 0


if __name__ == "__main__":
    # Minimal runner so the file works without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
