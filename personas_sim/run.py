"""
Run the experiment: for each survey question in config.QUESTIONS, compare five
methods at producing the answer distribution and measure each against the real
ground truth.

Methods
-------
  baseline     : ask the model with NO persona (null model).
  demographic  : 100 personas matching UK 16+ demographics, factual prompt.
  rich         : same demographics, fuller first-person backstory prompt.
  network      : multi-agent discussion. Round 1: each persona writes ONE short
                 opinion (free text). Round 2: each persona sees their 5-person
                 group's opinions and then picks a letter. Models social
                 influence rather than treating personas as isolated atoms.
  elicited     : verbalized sampling. Each persona reports a PROBABILITY over
                 every option instead of one hard letter; we average the soft
                 per-persona distributions. Recovers within-person uncertainty
                 and resists mode collapse (the dominant synthetic-persona
                 failure). Uses the same personas as `rich`, so the elicitation
                 format is the only variable.

All persona prompts carry an anti-sycophancy instruction (see personas.py),
another first-principles lever against mode collapse.

The methods are named implementations of specific results (silicon sampling,
verbalized sampling, generative agents, ...); SOURCES.md maps each choice to
its paper.

Reported
--------
- Per-question distribution accuracy (= 1 - TVD, %), JSD, and PEAK (modal-bucket
  share) -- peak vs the ground-truth peak surfaces mode collapse directly.
- Averaged distribution accuracy across all questions per method.
- Response consistency on the first question (within-persona modal-agreement
  across 5 rephrasings) for the hard-vote persona methods.
- Age-gradient sub-group check on Q1: younger people are really MORE concerned
  about climate; a method that only hit the marginal by mode-collapse will show
  a flat (or wrong-signed) gradient. This is the test that distinguishes a real
  result from a lucky marginal.

Usage:
  USE_MOCK=1 python -m personas_sim.run          # offline, no API key
  USE_MOCK=0 python -m personas_sim.run          # real model (needs API key)

Outputs: prints tables; writes results.json and comparison.png.
"""

import json
import os
import time

from .config import QUESTIONS
from .llm import (
    ask_persona, parse_letter, parse_distribution, USE_MOCK, MODEL,
)
from .personas import (
    build_personas, question_block, elicitation_prompt, opinion_prompt,
    question_block_after_discussion, make_groups,
)
from .evaluate import evaluate, evaluate_distribution, valid_letters, peak
from .consistency import measure_consistency
from .diagnostics import (
    age_gradient, persona_dispersion, axis_gradient,
    CONCERNED_PARTIES, SKEPTIC_PARTIES, HIGH_OPENNESS, LOW_OPENNESS,
    CONCERNED_VALUES, SKEPTIC_VALUES,
)

# Psychographic axes for the Q1-only extension, PER QUESTION. Each adds ONE
# sentence to the demographic prompt and is validated by a paired gradient
# (high- vs low-stance group) with a declared `expected` direction. The point
# of making this per-question: the dominant axis is TOPIC-DEPENDENT. For climate
# all three are expected positive; for AI, Openness is the pre-registered
# winner (openness to new technology) while politics/values are EXPLORATORY (no
# established AI direction -> reported, but no pass/fail claim).
_AX_POLITICAL = {"method": "psychographic", "label": "political", "attr": "affiliation",
                 "high": CONCERNED_PARTIES, "low": SKEPTIC_PARTIES, "adds": "political identity"}
_AX_OPENNESS = {"method": "openness", "label": "openness", "attr": "openness",
                "high": HIGH_OPENNESS, "low": LOW_OPENNESS, "adds": "Big Five openness"}
_AX_VALUES = {"method": "values", "label": "values", "attr": "values",
              "high": CONCERNED_VALUES, "low": SKEPTIC_VALUES, "adds": "Schwartz dominant value"}

AXES_BY_QID = {
    "personal_importance": [   # climate: all three expected positive
        {**_AX_POLITICAL, "expected": "positive"},
        {**_AX_OPENNESS,  "expected": "positive"},
        {**_AX_VALUES,    "expected": "positive"},
    ],
    "ai_llm_benefit": [        # AI: openness is the expected winner; rest exploratory
        {**_AX_OPENNESS,  "expected": "positive"},
        {**_AX_POLITICAL, "expected": "exploratory"},
        {**_AX_VALUES,    "expected": "exploratory"},
    ],
}

# N is overridable via env so a quick smoke-run (e.g. N=20) does not require
# editing code. Full reportable run is N=100 per the brief.
N = int(os.environ.get("N", "100"))
SEED = 0
GROUP_SIZE = 5                # network method: discussion-group size
OPINION_TOKENS = 80           # max_tokens for network round-1 opinions
ELICIT_TOKENS = 60            # max_tokens for elicited distributions
CONSISTENCY_SAMPLE = 10       # personas sampled (per method) for consistency
HUMAN_CEILING = 0.91          # Stanford intra-respondent repeat-rate ceiling
                              # (primary source: Park et al. 2024, "Generative
                              # Agent Simulations of 1,000 People" -- SOURCES.md)

METHODS = ("baseline", "demographic", "rich", "network", "elicited")
PERSONA_METHODS = ("demographic", "rich", "network", "elicited")
HARDVOTE_METHODS = ("demographic", "rich", "network")   # consistency applies
COLORS = ["#888888", "#33aa77", "#2277cc", "#cc6633", "#9944bb"]


# --- collectors -------------------------------------------------------------

def _collect_letters(personas, question):
    """Single-pass hard vote: each persona is asked the multiple-choice
    question and we parse one letter. Returns a list of letters (or None)."""
    qblock = question_block(question)
    valid = valid_letters(question)
    out = []
    for i, p in enumerate(personas):
        raw = ask_persona(p["system_prompt"], qblock)
        out.append(parse_letter(raw, valid))
        if i % 25 == 24:
            time.sleep(0.2)
    return out


def _collect_distributions(personas, question):
    """Elicited method: each persona reports a probability over every option.
    Returns a list of per-persona distributions (dict) or None for failures."""
    eprompt = elicitation_prompt(question)
    valid = valid_letters(question)
    out = []
    for i, p in enumerate(personas):
        raw = ask_persona(p["system_prompt"], eprompt,
                          max_tokens=ELICIT_TOKENS, mock_kind="distribution")
        out.append(parse_distribution(raw, valid))
        if i % 25 == 24:
            time.sleep(0.2)
    return out


def _mean_distribution(dists, valid):
    """Average the non-None per-persona distributions; return (mean, unparse)."""
    good = [d for d in dists if d is not None]
    if not good:
        return {k: 0.0 for k in valid}, 1.0
    mean = {k: sum(d[k] for d in good) / len(good) for k in valid}
    return mean, 1.0 - len(good) / len(dists)


def _run_network(personas, question):
    """round 1 -- each persona writes ONE short opinion (free text);
       round 2 -- each persona sees their group's other opinions, then votes."""
    valid = valid_letters(question)
    op_prompt = opinion_prompt(question)

    opinions = []
    for i, p in enumerate(personas):
        raw = ask_persona(p["system_prompt"], op_prompt,
                          max_tokens=OPINION_TOKENS, mock_kind="opinion")
        opinions.append(raw)
        if i % 25 == 24:
            time.sleep(0.2)

    groups = make_groups(len(personas), GROUP_SIZE, seed=SEED)
    answers = [None] * len(personas)
    for g_i, group in enumerate(groups):
        for idx in group:
            peers = [opinions[j] for j in group if j != idx]
            user_msg = question_block_after_discussion(question, peers)
            raw = ask_persona(personas[idx]["system_prompt"], user_msg)
            answers[idx] = parse_letter(raw, valid)
        if g_i % 5 == 4:
            time.sleep(0.2)
    return answers


# --- main -------------------------------------------------------------------

def _banner():
    """Print which model produced this output, so a pasted result is never
    ambiguous about whether it was mock/Haiku/Sonnet."""
    src = "MOCK (no API — illustrative only)" if USE_MOCK else f"model = {MODEL}"
    print(f"\n[run config] {src}   N={N}   seed={SEED}", flush=True)


def run():
    _banner()
    # Fast path: just the psychographic extension on Q1 (a cheap ~2*N-call run
    # that demonstrates the creative addition without re-running the full
    # 5-method suite). The main results stay whatever the last full run wrote.
    if os.environ.get("EXT_ONLY"):
        return _run_extension_only()

    # Build the persona pool ONCE per method and re-use across questions, so
    # method comparisons aren't contaminated by demographic sampling noise.
    # `elicited` shares `rich`'s persona text (same seed) -> only the
    # elicitation format differs.
    persona_pool = {m: build_personas(N, method=m, seed=SEED)
                    for m in PERSONA_METHODS}

    per_question = {}
    raw_first = None
    for question in QUESTIONS:
        evals, raw = _run_one_question(question, persona_pool)
        per_question[question["id"]] = evals
        if raw_first is None:
            raw_first = raw   # keep Q1's per-persona answers for the gradient

    # Consistency on the first question only, for the hard-vote methods.
    first_q = QUESTIONS[0]
    consistency = {
        m: measure_consistency(persona_pool[m], first_q,
                               n_personas=CONSISTENCY_SAMPLE)["score"]
        for m in HARDVOTE_METHODS
    }

    # Age-gradient sub-group check on Q1 for every persona-based method.
    gradients = {
        m: age_gradient(persona_pool[m], raw_first[m], first_q)
        for m in PERSONA_METHODS
    }

    # Persona-dispersion check on Q1 for `elicited`: are the soft per-persona
    # distributions actually different, or is the model reciting one aggregate?
    dispersion = persona_dispersion(raw_first["elicited"])

    # Q1-ONLY extension: the psychographic axis methods (+ paired gradients).
    # Kept out of the main 5-method tables so the headline results are
    # unchanged; reported in its own block.
    ext = _axes_extension(
        first_q, persona_pool["demographic"], raw_first["demographic"])

    averaged = _average_by_topic(per_question)

    _report(per_question, averaged, consistency, gradients, dispersion, ext)
    _save(per_question, averaged, consistency, gradients, dispersion, ext)
    return {"per_question": per_question, "averaged": averaged,
            "consistency": consistency, "age_gradient_q1": gradients,
            "elicited_dispersion_q1": dispersion,
            "axes_q1": ext}


def _target_question():
    """The question the psychographic-axes extension runs on. Defaults to the
    first question; override with EXT_QUESTION=<id> (e.g. ai_llm_benefit)."""
    qid = os.environ.get("EXT_QUESTION", QUESTIONS[0]["id"])
    for q in QUESTIONS:
        if q["id"] == qid:
            return q
    raise ValueError(f"EXT_QUESTION={qid!r} not in QUESTIONS")


def _axes_extension(question, demo_personas, demo_answers):
    """Run each psychographic axis method for `question` and, for each, compute
    its paired gradient AND the `demographic` control gradient (same split, axis
    NOT in the prompt -> should be flat). The axis set + each axis's expected
    direction come from AXES_BY_QID, so the same machinery serves climate
    (all expected positive) and AI (openness positive; politics/values
    exploratory). `demo_personas`/`demo_answers` are reused so the control costs
    no extra calls. An optional AXIS env var restricts to one axis."""
    only = os.environ.get("AXIS")
    axes = AXES_BY_QID.get(question["id"], [])
    out = {"demographic_eval": evaluate(demo_answers, question), "axes": {}}
    for ax in axes:
        if only and only not in (ax["label"], ax["method"]):
            continue
        personas = build_personas(N, method=ax["method"], seed=SEED)
        answers = _collect_letters(personas, question)
        out["axes"][ax["label"]] = {
            "adds": ax["adds"],
            "expected": ax["expected"],
            "eval": evaluate(answers, question),
            "gradient": axis_gradient(
                personas, answers, question, ax["attr"], ax["high"], ax["low"],
                expected=ax["expected"]),
            "control_gradient": axis_gradient(
                demo_personas, demo_answers, question, ax["attr"], ax["high"],
                ax["low"], expected=ax["expected"]),
        }

    # The composite ("kitchen-sink") persona stacks ALL axes at once. Only run it
    # when not filtering to a single axis. We score its accuracy and check that
    # every axis gradient still survives in the combined prompt.
    if not only and axes:
        comp_p = build_personas(N, method="composite", seed=SEED)
        comp_a = _collect_letters(comp_p, question)
        out["composite"] = {
            "eval": evaluate(comp_a, question),
            "gradients": {
                ax["label"]: axis_gradient(
                    comp_p, comp_a, question, ax["attr"], ax["high"], ax["low"],
                    expected=ax["expected"])
                for ax in axes
            },
        }
    return out


def _run_extension_only():
    """EXT_ONLY fast path: build the demographic control + each axis method for
    the target question (EXT_QUESTION, default Q1), report, and write a SEPARATE
    results_psychographic.json so the main results.json is untouched.
    Cost ~ (1 + n_axes + composite) * N calls."""
    q = _target_question()
    demo_personas = build_personas(N, method="demographic", seed=SEED)
    demo_answers = _collect_letters(demo_personas, q)
    ext = _axes_extension(q, demo_personas, demo_answers)

    print(f"\n(EXT_ONLY: psychographic axes on {q['id']} only, N={N})")
    _report_extension(q, ext)
    with open("results_psychographic.json", "w") as f:
        json.dump({"question": q["id"], "axes_q1": ext}, f, indent=2)
    print("wrote results_psychographic.json")
    return {"axes_q1": ext}


def _run_one_question(question, persona_pool):
    """Return (evals, raw) for one question, where
       evals = {method: evaluation-dict}
       raw   = {method: per-persona answers}  (letters, or dists for elicited)."""
    evals, raw = {}, {}
    valid = valid_letters(question)

    # baseline -- no persona
    qblock = question_block(question)
    base = [parse_letter(ask_persona("", qblock), valid) for _ in range(N)]
    evals["baseline"] = evaluate(base, question)
    raw["baseline"] = base

    for m in ("demographic", "rich"):
        ans = _collect_letters(persona_pool[m], question)
        evals[m] = evaluate(ans, question)
        raw[m] = ans

    net = _run_network(persona_pool["network"], question)
    evals["network"] = evaluate(net, question)
    raw["network"] = net

    dists = _collect_distributions(persona_pool["elicited"], question)
    mean, unparse = _mean_distribution(dists, valid)
    evals["elicited"] = evaluate_distribution(mean, question, unparse, len(dists))
    raw["elicited"] = dists

    return evals, raw


def _topics():
    """Distinct topic tags, in first-seen order (climate first, then ai)."""
    seen = []
    for q in QUESTIONS:
        t = q.get("topic", "other")
        if t not in seen:
            seen.append(t)
    return seen


def _average_by_topic(per_question):
    """Mean distribution accuracy per method, grouped BY TOPIC, so the climate
    average stays comparable to earlier runs and the AI generality test is
    reported on its own rather than blended in. Returns {topic: {method: avg}}."""
    out = {}
    for t in _topics():
        qs = [q for q in QUESTIONS if q.get("topic", "other") == t]
        out[t] = {}
        for m in METHODS:
            accs = [per_question[q["id"]][m]["distribution_accuracy"] for q in qs]
            out[t][m] = sum(accs) / len(accs) if accs else 0.0
    return out


# --- reporting --------------------------------------------------------------

def _report(per_question, averaged, consistency, gradients, dispersion, ext=None):
    print(f"\nHuman-replication ceiling (Stanford intra-respondent): "
          f"{HUMAN_CEILING:.0%} distribution accuracy\n")

    for question in QUESTIONS:
        qid = question["id"]
        letters = valid_letters(question)
        gt = question["ground_truth"]
        gt_peak = peak(gt)
        print(f"=== {qid}  ({question['source']}) ===")
        print(f"    \"{question['text']}\"")
        gt_s = "  ".join(f"{k}={gt[k]:.2f}" for k in letters)
        print(f"    ground truth:   {gt_s}   (peak {100*gt_peak:.0f}%)")
        print("    {:<13} {:>9} {:>6} {:>6} {:>9}   distribution".format(
            "method", "dist.acc%", "JSD", "peak%", "unparse%"))
        print("    " + "-" * 84)
        for m in METHODS:
            r = per_question[qid][m]
            dist = "  ".join(f"{k}={r['distribution'][k]:.2f}" for k in letters)
            print("    {:<13} {:>8.1f}% {:>6.3f} {:>5.0f}% {:>8.1f}%   {}".format(
                m,
                100 * r["distribution_accuracy"],
                r["jsd"],
                100 * r["peak"],
                100 * r["unparseable_rate"],
                dist,
            ))
        print()

    print("=== AVERAGED distribution accuracy, by topic ===")
    first_qid = QUESTIONS[0]["id"]
    for topic, per_m in averaged.items():
        qs = [q for q in QUESTIONS if q.get("topic", "other") == topic]
        # consistency was measured on the first question only; show it for the
        # topic that contains it.
        show_cons = any(q["id"] == first_qid for q in qs)
        print(f"  topic: {topic} ({len(qs)} question{'s' if len(qs) != 1 else ''})")
        hdr = "    {:<13} {:>9}".format("method", "dist.acc%")
        if show_cons:
            hdr += "   {:>9}".format("consist%")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for m in METHODS:
            line = "    {:<13} {:>8.1f}%".format(m, 100 * per_m[m])
            if show_cons:
                cons = consistency.get(m)
                line += "   " + (f"{100 * cons:>7.1f}%" if cons is not None
                                 else "      --")
            print(line)
        best = max(per_m, key=per_m.get)
        print(f"    best: {best!r} ({100 * per_m[best]:.1f}%)\n")

    # Age-gradient sub-group check
    fq = QUESTIONS[0]["id"]
    print(f"\n=== Age-gradient check on Q1 ({fq}) ===")
    print("    younger people are really MORE concerned -> expect a POSITIVE "
          "gradient")
    print("    {:<13} {:>7} {:>7} {:>10} {:>12}".format(
        "method", "young", "old", "gradient", "as_expected"))
    print("    " + "-" * 54)
    for m in PERSONA_METHODS:
        g = gradients[m]
        if g["gradient"] is None:
            print("    {:<13} {:>7} {:>7} {:>10} {:>12}".format(
                m, "--", "--", "--", "n/a"))
            continue
        print("    {:<13} {:>7.2f} {:>7.2f} {:>+10.2f} {:>12}".format(
            m, g["young_mean"], g["old_mean"], g["gradient"],
            "YES" if g["as_expected"] else "no"))
    print("    (concern score: higher = more concerned; gradient = young - old)")

    # Elicited persona-dispersion: is `elicited` simulating or reciting?
    d = dispersion.get("mean_pairwise_tvd")
    print(f"\n=== Elicited persona-dispersion check on Q1 ===")
    if d is None:
        print("    not enough parseable distributions to compute")
    else:
        print(f"    mean pairwise TVD between personas' elicited distributions: "
              f"{d:.3f}  (n={dispersion['n']})")
        print("    ~0 => personas near-identical: a perfect marginal is "
              "RECITATION of a known")
        print("         aggregate, not simulation of distinct people. "
              "Higher => personas individuate.")

    if ext is not None:
        _report_extension(QUESTIONS[0], ext)


def _grad_tag(g):
    """Pass/fail tag honouring the axis's declared expectation: 'exploratory'
    when there is no expected direction, 'n/a' when ungradable."""
    if g.get("gradient") is None:
        return "n/a"
    if g.get("as_expected") is None:
        return "exploratory"
    return "YES" if g["as_expected"] else "no"


def _report_extension(first_q, ext):
    """Print the psychographic-axes extension for one question: each axis
    method's accuracy vs the `demographic` control, then the paired gradient
    checks (axis in-prompt vs demographic control) that show the axis is doing
    real work, not leaking via correlated demographics."""
    letters = valid_letters(first_q)
    demo = ext["demographic_eval"]
    print(f"\n=== Extension: psychographic axes ({first_q['id']}) ===")
    print("    each axis = the demographic prompt + ONE extra sentence; "
          "demographic is the control")
    print("    {:<16} {:>9} {:>6} {:>6}   distribution".format(
        "method", "dist.acc%", "JSD", "peak%"))
    print("    " + "-" * 84)
    demo_dist = "  ".join(f"{k}={demo['distribution'][k]:.2f}" for k in letters)
    print("    {:<16} {:>8.1f}% {:>6.3f} {:>5.0f}%   {}".format(
        "demographic", 100 * demo["distribution_accuracy"], demo["jsd"],
        100 * demo["peak"], demo_dist))
    for label, a in ext["axes"].items():
        r = a["eval"]
        dist = "  ".join(f"{k}={r['distribution'][k]:.2f}" for k in letters)
        print("    {:<16} {:>8.1f}% {:>6.3f} {:>5.0f}%   {}".format(
            label, 100 * r["distribution_accuracy"], r["jsd"],
            100 * r["peak"], dist))
    if "composite" in ext:
        r = ext["composite"]["eval"]
        dist = "  ".join(f"{k}={r['distribution'][k]:.2f}" for k in letters)
        print("    {:<16} {:>8.1f}% {:>6.3f} {:>5.0f}%   {}".format(
            "composite", 100 * r["distribution_accuracy"], r["jsd"],
            100 * r["peak"], dist))
        print("    (composite = demographic + all 3 axis sentences at once)")

    print(f"\n=== Paired gradient checks ({first_q['id']}) ===")
    print("    axis in-prompt should match its expected direction; the "
          "demographic control should be ~flat")
    print("    {:<14} {:>11} {:>11} {:>13}".format(
        "axis", "in-prompt", "control", "as_expected"))
    print("    " + "-" * 52)
    for label, a in ext["axes"].items():
        g, c = a["gradient"], a["control_gradient"]
        gv = f"{g['gradient']:+.2f}" if g["gradient"] is not None else "n/a"
        cv = f"{c['gradient']:+.2f}" if c["gradient"] is not None else "n/a"
        print("    {:<14} {:>11} {:>11} {:>13}".format(label, gv, cv, _grad_tag(g)))
    print("    (gradient = high-stance group minus low-stance group; "
          "'exploratory' = no pre-set direction for this topic)")

    if "composite" in ext:
        print(f"\n=== Composite persona: do the gradients survive together? "
              f"({first_q['id']}) ===")
        print("    all axes in one prompt; each gradient should keep its "
              "expected direction")
        print("    {:<14} {:>11} {:>13}".format("axis", "gradient", "as_expected"))
        print("    " + "-" * 40)
        for label, g in ext["composite"]["gradients"].items():
            gv = f"{g['gradient']:+.2f}" if g["gradient"] is not None else "n/a"
            print("    {:<14} {:>11} {:>13}".format(label, gv, _grad_tag(g)))


def _save(per_question, averaged, consistency, gradients, dispersion, ext=None):
    out = {
        "human_ceiling": HUMAN_CEILING,
        "questions": [
            {"id": q["id"], "text": q["text"], "source": q["source"],
             "options": q["options"], "ground_truth": q["ground_truth"]}
            for q in QUESTIONS
        ],
        "per_question": per_question,
        "averaged_distribution_accuracy_by_topic": averaged,
        "consistency_first_question": consistency,
        "age_gradient_q1": gradients,
        "elicited_dispersion_q1": dispersion,
        "axes_q1": ext,
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    try:
        _plot(per_question, averaged)
    except Exception as e:
        print(f"(skipped plot: {e})")


def _plot(per_question, averaged):
    """One subplot per question (each method's distribution vs truth) plus an
    averaged-accuracy summary with the human-ceiling line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_q = len(QUESTIONS)
    fig, axes = plt.subplots(n_q + 1, 1, figsize=(11, 3.4 * (n_q + 1)))
    if n_q + 1 == 1:
        axes = [axes]

    for ax, question in zip(axes[:n_q], QUESTIONS):
        qid = question["id"]
        letters = valid_letters(question)
        gt = question["ground_truth"]
        x = np.arange(len(letters))
        width = 0.8 / (len(METHODS) + 1)

        ax.bar(x - 0.4 + width / 2, [gt[k] for k in letters], width,
               label="Ground truth", color="black")
        for i, m in enumerate(METHODS):
            d = per_question[qid][m]["distribution"]
            acc = per_question[qid][m]["distribution_accuracy"]
            ax.bar(x - 0.4 + width * (i + 1.5),
                   [d[k] for k in letters], width,
                   label=f"{m} (acc {100*acc:.0f}%)", color=COLORS[i])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{k}\n{question['options'][k].split()[0]}" for k in letters])
        ax.set_ylabel("proportion")
        ax.set_title(f"{qid}: {question['text']}")
        ax.legend(fontsize=8, loc="upper right")

    # Bottom subplot: averaged accuracy for the FIRST topic (climate), the
    # headline set; `averaged` is now {topic: {method: avg}}.
    ax = axes[-1]
    topic0 = next(iter(averaged))
    avg0 = averaged[topic0]
    n_topic_q = sum(1 for q in QUESTIONS if q.get("topic", "other") == topic0)
    methods = list(METHODS)
    accs = [100 * avg0[m] for m in methods]
    ax.bar(methods, accs, color=COLORS[:len(methods)])
    ax.axhline(100 * HUMAN_CEILING, color="black", linestyle="--",
               label=f"Human ceiling ({100*HUMAN_CEILING:.0f}%)")
    ax.set_ylabel("avg distribution accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Averaged across {n_topic_q} {topic0} questions")
    for m, a in zip(methods, accs):
        ax.text(m, a + 1, f"{a:.0f}%", ha="center", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)
    print("wrote comparison.png and results.json")


if __name__ == "__main__":
    run()
