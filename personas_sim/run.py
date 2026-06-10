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
from .diagnostics import age_gradient, persona_dispersion, political_gradient

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

    # Q1-ONLY extension: the `psychographic` method (+ political gradient).
    # Kept out of the main 5-method tables so the headline results are
    # unchanged; reported in its own block.
    ext = _psychographic_extension(
        first_q, persona_pool["demographic"], raw_first["demographic"])

    averaged = _average_accuracy(per_question)

    _report(per_question, averaged, consistency, gradients, dispersion, ext)
    _save(per_question, averaged, consistency, gradients, dispersion, ext)
    return {"per_question": per_question, "averaged": averaged,
            "consistency": consistency, "age_gradient_q1": gradients,
            "elicited_dispersion_q1": dispersion,
            "psychographic_q1": ext}


def _psychographic_extension(first_q, demo_personas, demo_answers):
    """Run the `psychographic` method on Q1 and compute the political-gradient
    check for it AND for `demographic` (the control). Returns a dict used by
    the report + JSON. `demo_personas`/`demo_answers` are reused from the main
    run so the control costs no extra calls."""
    psy_personas = build_personas(N, method="psychographic", seed=SEED)
    psy_answers = _collect_letters(psy_personas, first_q)
    return {
        "demographic_eval": evaluate(demo_answers, first_q),
        "psychographic_eval": evaluate(psy_answers, first_q),
        "political_gradient": {
            "demographic": political_gradient(demo_personas, demo_answers, first_q),
            "psychographic": political_gradient(psy_personas, psy_answers, first_q),
        },
    }


def _run_extension_only():
    """EXT_ONLY fast path: build demographic + psychographic pools, collect Q1
    answers for both (~2*N calls), report the extension, and write a SEPARATE
    results_psychographic.json so the main results.json is untouched."""
    first_q = QUESTIONS[0]
    demo_personas = build_personas(N, method="demographic", seed=SEED)
    demo_answers = _collect_letters(demo_personas, first_q)
    ext = _psychographic_extension(first_q, demo_personas, demo_answers)

    print(f"\n(EXT_ONLY: psychographic extension on Q1 only, N={N})")
    _report_extension(first_q, ext)
    with open("results_psychographic.json", "w") as f:
        json.dump({"question": first_q["id"], "psychographic_q1": ext}, f, indent=2)
    print("wrote results_psychographic.json")
    return {"psychographic_q1": ext}


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


def _average_accuracy(per_question):
    """Mean distribution accuracy per method across all questions."""
    out = {}
    for m in METHODS:
        accs = [per_question[q["id"]][m]["distribution_accuracy"]
                for q in QUESTIONS]
        out[m] = sum(accs) / len(accs) if accs else 0.0
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

    print("=== AVERAGED across questions ===")
    print("    {:<13} {:>9}   {:>9}".format("method", "dist.acc%", "consist%"))
    print("    " + "-" * 40)
    for m in METHODS:
        cons = consistency.get(m)
        cons_s = f"{100 * cons:>7.1f}%" if cons is not None else "      --"
        print("    {:<13} {:>8.1f}%   {}".format(m, 100 * averaged[m], cons_s))
    best = max(averaged, key=averaged.get)
    print(f"\n    Best avg accuracy: {best!r} ({100*averaged[best]:.1f}%, "
          f"ceiling {HUMAN_CEILING:.0%})")

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


def _report_extension(first_q, ext):
    """Print the Q1-only `psychographic` extension: accuracy vs the
    `demographic` control, and the political-gradient money shot."""
    letters = valid_letters(first_q)
    print(f"\n=== Extension: psychographic method (Q1 only: {first_q['id']}) ===")
    print("    demographic (no politics) vs psychographic (+ political identity)")
    print("    {:<14} {:>9} {:>6} {:>6}   distribution".format(
        "method", "dist.acc%", "JSD", "peak%"))
    print("    " + "-" * 80)
    for name, key in (("demographic", "demographic_eval"),
                      ("psychographic", "psychographic_eval")):
        r = ext[key]
        dist = "  ".join(f"{k}={r['distribution'][k]:.2f}" for k in letters)
        print("    {:<14} {:>8.1f}% {:>6.3f} {:>5.0f}%   {}".format(
            name, 100 * r["distribution_accuracy"], r["jsd"],
            100 * r["peak"], dist))

    print(f"\n=== Political-gradient check on Q1 ({first_q['id']}) ===")
    print("    climate-progressive-party voters are really MORE concerned ->")
    print("    expect POSITIVE for psychographic, ~flat for demographic (control)")
    print("    {:<14} {:>9} {:>9} {:>10} {:>12}".format(
        "method", "progress.", "skeptic", "gradient", "as_expected"))
    print("    " + "-" * 58)
    for m in ("demographic", "psychographic"):
        g = ext["political_gradient"][m]
        if g["gradient"] is None:
            print("    {:<14} {:>9} {:>9} {:>10} {:>12}".format(
                m, "--", "--", "--", "n/a"))
            continue
        tag = "YES" if g["as_expected"] else "no"
        if m == "demographic":
            tag += " (ctrl)"
        print("    {:<14} {:>9.2f} {:>9.2f} {:>+10.2f} {:>12}".format(
            m, g["green_mean"], g["right_mean"], g["gradient"], tag))
    print("    (concern score: higher = more concerned; "
          "gradient = progressive - skeptic)")


def _save(per_question, averaged, consistency, gradients, dispersion, ext=None):
    out = {
        "human_ceiling": HUMAN_CEILING,
        "questions": [
            {"id": q["id"], "text": q["text"], "source": q["source"],
             "options": q["options"], "ground_truth": q["ground_truth"]}
            for q in QUESTIONS
        ],
        "per_question": per_question,
        "averaged_distribution_accuracy": averaged,
        "consistency_first_question": consistency,
        "age_gradient_q1": gradients,
        "elicited_dispersion_q1": dispersion,
        "psychographic_q1": ext,
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

    ax = axes[-1]
    methods = list(METHODS)
    accs = [100 * averaged[m] for m in methods]
    ax.bar(methods, accs, color=COLORS[:len(methods)])
    ax.axhline(100 * HUMAN_CEILING, color="black", linestyle="--",
               label=f"Human ceiling ({100*HUMAN_CEILING:.0f}%)")
    ax.set_ylabel("avg distribution accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Averaged across {n_q} questions")
    for m, a in zip(methods, accs):
        ax.text(m, a + 1, f"{a:.0f}%", ha="center", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)
    print("wrote comparison.png and results.json")


if __name__ == "__main__":
    run()
