"""
Response consistency: ask the SAME persona the SAME question in several
different wordings and measure how often they give the same answer.

Mirrors the headline metric in the AS Jan-2026 eval report: a faithful persona
should be stable to paraphrase, not just produce a plausible one-shot answer.
Distribution accuracy alone can be right for the wrong reason -- a population
of incoherent personas can still average out to the target. Consistency
catches that failure mode.

Metric: per-persona modal-agreement rate = (count of answers matching that
persona's modal answer) / (number of rephrasings). Mean across personas is
the reported score; 1.0 means every persona always gave the same letter
regardless of wording.

We run consistency on the FIRST question only by default (it is the most
robustly verified ground truth) -- consistency is a property of the persona's
stability, and one question is enough to demonstrate the metric without
multiplying the call budget.
"""

from collections import Counter

from .llm import ask_persona, parse_letter
from .personas import question_block


def measure_consistency(personas, question, n_personas: int = 10):
    """Return {'score', 'per_persona', 'n', 'rephrasings'}. Subsamples
    personas because every persona costs len(rephrasings) calls."""
    rephrasings = question.get("rephrasings") or [question["text"]]
    valid = list(question["options"].keys())
    sample = personas[:n_personas]

    per_persona = []
    for p in sample:
        answers = []
        for q in rephrasings:
            raw = ask_persona(p["system_prompt"], question_block(question, q))
            answers.append(parse_letter(raw, valid))
        non_null = [a for a in answers if a is not None]
        if not non_null:
            per_persona.append(0.0)
            continue
        modal_count = Counter(non_null).most_common(1)[0][1]
        per_persona.append(modal_count / len(rephrasings))

    score = sum(per_persona) / len(per_persona) if per_persona else 0.0
    return {"score": score, "per_persona": per_persona, "n": len(sample),
            "rephrasings": len(rephrasings)}
