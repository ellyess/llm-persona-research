"""
The only part that touches an LLM. Treated as a black box: text in, answer out.

- ask_persona():       send a persona (system prompt) + question block, get text.
- parse_letter():      turn a messy reply into a valid option letter, or None.
- parse_distribution(): turn an elicited "A: 20, B: 35, ..." reply into a
                        normalised dict over the valid letters, or None.

Runs in MOCK mode by default (no API key, fake but non-uniform answers) so the
whole pipeline is testable offline. Set USE_MOCK=False and an ANTHROPIC_API_KEY
env var to use the real model.
"""

import os
import re
import random
import time

USE_MOCK = os.environ.get("USE_MOCK", "1") == "1"
MODEL = os.environ.get("MODEL", "claude-haiku-4-5")

# Per-call retry budget for transient errors (429 rate-limit, 5xx, network).
# Exponential backoff starting at BACKOFF_BASE seconds, doubling each retry.
MAX_RETRIES = 6
BACKOFF_BASE = 4.0


def ask_persona(persona_text: str, question_block: str, max_tokens: int = 10,
                mock_kind: str = "letter") -> str:
    """Send persona + question to the model; return the raw reply text.

    `max_tokens` is small by default (we just want a letter); the network
    method's opinion round and the elicited method's distribution need more.
    `mock_kind` ("letter" | "opinion" | "distribution") only affects the
    offline mock -- it is ignored when calling the real API.

    Retries on rate-limit / transient errors with exponential backoff so a
    long run doesn't lose all completed work to a single 429."""
    if USE_MOCK:
        return _mock_reply(persona_text, mock_kind)

    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=persona_text,
                messages=[{"role": "user", "content": question_block}],
            )
            return resp.content[0].text
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            status = getattr(e, "status_code", None)
            retriable = isinstance(e, (anthropic.RateLimitError,
                                       anthropic.APIConnectionError)) or (
                status is not None and status >= 500)
            if not retriable or attempt == MAX_RETRIES - 1:
                raise
            sleep_s = BACKOFF_BASE * (2 ** attempt)
            print(f"  [retry {attempt+1}/{MAX_RETRIES} after {sleep_s:.0f}s: "
                  f"{type(e).__name__}]", flush=True)
            time.sleep(sleep_s)


def parse_letter(reply: str, valid_letters):
    """First valid standalone option letter in `reply`, else None. Uses
    negative lookarounds so a letter inside a word ('A' in 'Answer') is
    not matched -- only a standalone letter counts.

    Note: hard-vote letter parsing is exposed to ordering/labeling bias (models
    lean toward 'A'); see Domínguez-Olmedo et al. 2024 in SOURCES.md. The
    soft-distribution (`elicited`) path and the peak/entropy/dispersion
    diagnostics exist partly to surface that failure mode."""
    if not reply:
        return None
    for letter in valid_letters:
        if re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", reply, re.IGNORECASE):
            return letter.upper()
    return None


def parse_distribution(reply: str, valid_letters):
    """Parse an elicited reply like 'A: 20, B: 35, C: 30, D: 10, E: 5' into a
    normalised dict {letter: prob}. Accepts ':', ')', '=' or '-' separators and
    ignores stray '%'. Returns None if no recognised number is found for any
    letter (treated as unparseable). Letters not mentioned default to 0; the
    result is renormalised to sum to 1."""
    if not reply:
        return None
    raw = {}
    for letter in valid_letters:
        m = re.search(rf"(?<![A-Za-z]){letter}\s*[:\)\=\-]\s*([0-9]*\.?[0-9]+)",
                      reply, re.IGNORECASE)
        if m:
            raw[letter] = float(m.group(1))
    if not raw:
        return None
    total = sum(raw.values())
    if total <= 0:
        return None
    return {k: raw.get(k, 0.0) / total for k in valid_letters}


# --- Mock model: makes the pipeline runnable with no API key ------------------
def _extract_politics(text: str):
    """Detect a party lean from the persona text. Only the `psychographic`
    prompt mentions a party, so `demographic` personas come back 'neutral' --
    which is exactly what makes the political-gradient control come out flat."""
    t = text.lower()
    if ("labour" in t or "liberal democrat" in t or "green party" in t
            or "nationalist party" in t):
        return "concerned"
    if "conservative" in t or "reform uk" in t:
        return "skeptic"
    return "neutral"


def _mock_weights(persona_text: str):
    """Weights over the 5-letter space. Age makes an age gradient visible;
    political lean (when present in the prompt) shifts climate-concern, so the
    mock demonstrates the `psychographic` method and its gradient offline.
    NOT meant to be realistic."""
    age = _extract_age(persona_text)
    if age is not None and age < 35:
        base = [0.30, 0.34, 0.24, 0.08, 0.04]    # younger: more "important"
    elif age is not None and age >= 60:
        base = [0.14, 0.24, 0.34, 0.18, 0.10]    # older: more spread
    else:
        base = [0.22, 0.30, 0.30, 0.12, 0.06]

    pol = _extract_politics(persona_text)
    if pol == "concerned":
        mult = [1.7, 1.3, 0.9, 0.6, 0.5]         # shift toward A/B (concern)
    elif pol == "skeptic":
        mult = [0.5, 0.8, 1.1, 1.5, 1.8]         # shift toward D/E (less)
    else:
        return base

    w = [b * m for b, m in zip(base, mult)]
    s = sum(w)
    return [x / s for x in w]


def _mock_reply(persona_text: str, kind: str) -> str:
    age = _extract_age(persona_text)
    letters = ["A", "B", "C", "D", "E"]
    weights = _mock_weights(persona_text)

    if kind == "opinion":
        if age is not None and age < 35:
            return random.choice([
                "Honestly it's the defining issue for my generation.",
                "I think about it more than I'd like, and I do worry.",
                "It feels urgent -- people my age can't really ignore it.",
            ])
        if age is not None and age >= 60:
            return random.choice([
                "I take it seriously but I trust the evidence will guide us.",
                "I worry less for me than for the grandchildren, really.",
                "It matters, though the news makes it harder to follow.",
            ])
        return random.choice([
            "I think it matters and I try to act on it where I can.",
            "It's serious, but I'm not in a position to do much about it.",
            "I care about it -- maybe not as much as I should.",
        ])

    if kind == "distribution":
        # Emit a noisy soft distribution centred on the same age-based weights.
        noisy = [max(0.0, w + random.uniform(-0.05, 0.05)) for w in weights]
        s = sum(noisy)
        pct = [round(100 * x / s) for x in noisy]
        return ", ".join(f"{L}: {p}" for L, p in zip(letters, pct))

    # default: single letter
    letter = random.choices(letters, weights=weights)[0]
    return random.choice([letter, f"{letter})", f"Answer: {letter}", f"I'd say {letter}."])


def _extract_age(text: str):
    m = re.search(r"\b(\d{2})\b", text)
    return int(m.group(1)) if m else None
