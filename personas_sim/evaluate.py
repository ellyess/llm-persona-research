"""
Evaluation: turn a method's output into a distribution and measure how faithful
it is to the real-world ground truth for ONE question.

Two entry points, because methods produce two shapes of output:
  - evaluate(answers, question)        : hard votes  -> distribution (most methods)
  - evaluate_distribution(dist, ...)   : an already-averaged soft distribution
                                         (the `elicited` / verbalized-sampling method)

Metrics:
  - distribution_accuracy = 1 - TVD, as a fraction ("what share of response
    mass landed in the right buckets"); the standard way to score an LM opinion
    distribution against a survey result.
  - JSD: symmetric, bounded [0,1] (log base 2).
  - peak: the modal bucket's share. Compared to the ground-truth peak, this
    surfaces MODE COLLAPSE -- a method whose peak is far above truth's has
    funnelled the population onto one option.
  - entropy: Shannon entropy (bits). Low entropy vs ground truth is the same
    mode-collapse signal from the spread side.
  - bootstrap_accuracy_ci: a percentile confidence interval for the accuracy,
    by resampling the personas with replacement, so method gaps can be read as
    real or as sampling noise.

Also reports the unparseable rate (failed answers) as a data-quality check.
"""

import math
import random


def valid_letters(question) -> list:
    return list(question["options"].keys())


def to_distribution(answers, question) -> dict:
    """List of letters (may include None) -> normalised distribution over the
    question's valid options. Unparseable answers are excluded and counted
    separately via unparseable_rate()."""
    letters = valid_letters(question)
    counts = {k: 0 for k in letters}
    valid = 0
    for a in answers:
        if a in counts:
            counts[a] += 1
            valid += 1
    if valid == 0:
        return {k: 0.0 for k in letters}
    return {k: counts[k] / valid for k in letters}


def unparseable_rate(answers) -> float:
    if not answers:
        return 0.0
    return sum(1 for a in answers if a is None) / len(answers)


def total_variation(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def distribution_accuracy(p: dict, q: dict) -> float:
    """1 - TVD, in [0,1]. Scoring LM opinion *distributions* against real survey
    data follows the OpinionQA methodology (Santurkar et al. 2023 -- see
    SOURCES.md)."""
    return 1.0 - total_variation(p, q)


def jensen_shannon(p: dict, q: dict) -> float:
    m = {k: 0.5 * (p[k] + q[k]) for k in p}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: dict, q: dict) -> float:
    total = 0.0
    for k in p:
        if p[k] > 0 and q[k] > 0:
            total += p[k] * math.log2(p[k] / q[k])
    return total


def entropy(p: dict) -> float:
    """Shannon entropy in bits. Lower = more peaked (mode-collapse signal)."""
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def peak(p: dict) -> float:
    """Modal bucket share. Higher than the ground-truth peak = mode collapse."""
    return max(p.values()) if p else 0.0


def _metrics(dist: dict, question: dict) -> dict:
    """Shared metric block for a finished distribution."""
    gt = question["ground_truth"]
    return {
        "distribution": dist,
        "distribution_accuracy": distribution_accuracy(dist, gt),
        "tvd": total_variation(dist, gt),
        "jsd": jensen_shannon(dist, gt),
        "peak": peak(dist),
        "entropy": entropy(dist),
    }


def evaluate(answers, question) -> dict:
    """Summary for a method that emits hard votes (one letter per persona)."""
    out = _metrics(to_distribution(answers, question), question)
    out["unparseable_rate"] = unparseable_rate(answers)
    out["n"] = len(answers)
    return out


def evaluate_distribution(dist: dict, question: dict,
                          unparseable: float = 0.0, n: int = 0) -> dict:
    """Summary for a method that emits an already-averaged soft distribution
    (the `elicited` method). `unparseable` is the share of personas whose
    distribution failed to parse and were excluded from the average."""
    out = _metrics(dist, question)
    out["unparseable_rate"] = unparseable
    out["n"] = n
    return out


def _mean_soft(dists, question) -> dict:
    """Average non-None per-persona distributions (mirrors the `elicited`
    method's aggregation in run.py)."""
    letters = valid_letters(question)
    good = [d for d in dists if d is not None]
    if not good:
        return {k: 0.0 for k in letters}
    return {k: sum(d[k] for d in good) / len(good) for k in letters}


def bootstrap_accuracy_ci(answers, question, n_boot=2000, seed=0, conf=0.95):
    """Percentile bootstrap confidence interval for distribution accuracy.

    `answers` is the per-persona output, EITHER a list of letters/None
    (hard-vote methods) OR a list of soft-distribution dicts/None (the
    `elicited` method). Resamples the personas with replacement `n_boot` times,
    recomputes the population distribution and its accuracy each time, and
    returns (lo, hi) at the given confidence level. The interval reflects
    persona sampling noise at this N; it does NOT capture model stochasticity
    across runs (that needs repeated full runs). Returns (None, None) if empty."""
    n = len(answers)
    if n == 0:
        return (None, None)
    gt = question["ground_truth"]
    soft = any(isinstance(a, dict) for a in answers)
    rng = random.Random(seed)
    accs = []
    for _ in range(n_boot):
        sample = [answers[rng.randrange(n)] for _ in range(n)]
        dist = _mean_soft(sample, question) if soft else to_distribution(sample, question)
        accs.append(distribution_accuracy(dist, gt))
    accs.sort()
    lo = accs[int((1 - conf) / 2 * n_boot)]
    hi = accs[min(n_boot - 1, int((1 + conf) / 2 * n_boot))]
    return (lo, hi)
