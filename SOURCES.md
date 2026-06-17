# Sources & references

Two things live here, kept in one place so the trail is auditable:

- **Part 1, Data & provenance:** every real-world *number* used in this project and
  where it comes from. The key property: **every real-world number comes from one
  publication** (Yale CCBM 2024), so the opinion frame and the demographic frame match
  by construction. There is no ONS dependency.
- **Part 2, References:** the academic *literature* behind the design choices. Most of
  the five methods are named implementations of a specific result; Part 2 maps each
  choice to its source. Inline `(Author Year, see SOURCES.md)` tags in the code point
  back to entries there.

---
---

# Part 1: Data & provenance

## 1. Source

**Climate Change in the British Mind, 2024**, Yale Program on Climate Change
Communication (YPCCC).

- Leiserowitz, A., Carman, J., Goddard, E., Verner, M., Rosenthal, S., &
  Marlon, J. (2025). *Climate Change in the British Mind, 2024.* Yale
  University, New Haven, CT.
- Report: https://climatecommunication.yale.edu/publications/climate-change-british-mind/
- PDF: https://climatecommunication.yale.edu/app/uploads/2025/01/climate-change-british-mind-a.pdf
- Frame: nationally representative survey of UK residents **aged 16+**,
  10,660 interviews, 7–13 November 2024, margin of error ±0.9pts at 95%.

---

## 2. Ground-truth opinion distributions

Used in [`personas_sim/config.py`](personas_sim/config.py) → `QUESTIONS`.
Each question's `source` field records its provenance at runtime.

| id | report item | item text | scale |
|----|-------------|-----------|-------|
| `personal_importance` | Q2.3 | How important is climate change to you personally? | 5-point |
| `worry` | Q2.1 | How worried are you about climate change? | 4-point |
| `harm_personally` | Q2.2 | How much will climate change harm you personally? | 5-point (incl. "Don't know") |

Mixed scales (4- and 5-point) are deliberate: they exercise the per-question
machinery and show the harness isn't hard-coded to one option count.

Percentages are the report's body-text / data-table figures; some columns sum
to 99–101 due to rounding. Appendix I holds exact unrounded values if needed.

---

## 3. Demographic distributions

Used in [`personas_sim/config.py`](personas_sim/config.py) → `DEMOGRAPHICS`.
Taken from **Appendix III "Sample Demographics"** of the same report.

### Why Appendix III beats stitching in ONS census data

- **Frame match by construction.** Appendix III *is* the composition of the
  16+ UK population whose opinion we're trying to reproduce. No recomputing
  16+ shares from whole-population census figures, no England-and-Wales-only
  compromise, no separate Scotland/NI tables. The demographic frame and the
  opinion frame are literally the same sample.
- **Single citable source.** One PDF backs every real-world number here.

### Weighted, not unweighted

Appendix III reports both an unweighted *n* and a **weighted %**. We sample
from the **weighted** column: the weighting is exactly what makes the raw
sample nationally representative, so it's the right target.

### Attributes available vs. used

`DEMOGRAPHICS` transcribes the **full** table faithfully (sex, generation,
education, income, region, ethnicity), including `prefer not to say` /
`don't know` buckets, for transparency.

Persona construction (`PROMPT_ATTRIBUTES` in
[`personas_sim/personas.py`](personas_sim/personas.py)) uses a subset:
**generation, sex, region, education, income**.

- **Ethnicity** is transcribed but *not* used in prompts: weak, poorly-evidenced
  link to climate opinion, and building a persona around it risks crude
  stereotyping.
- **`prefer not to say` / `don't know`** buckets are dropped and the remaining
  categories renormalised (`_clean_dist()`), because you can't anchor a
  coherent persona on "you are someone who declines to state their income."
  The raw buckets stay in `config.py` so the source remains faithfully
  recorded. Income loses ~24% of its mass to these buckets before
  renormalising, worth stating.

### Generation → age

Yale report **generation**, not age bands. Generations map cleanly onto the
climate-attitude age gradient and onto a representative 2024 age, which the
persona prompt needs. `GEN_AGE_RANGES` in `personas.py`:

| generation | 2024 age range used |
|------------|---------------------|
| Gen Z (1997 or later) | 16–27 (floored at the 16+ frame) |
| Millennials (1981–1996) | 28–43 |
| Gen X (1965–1980) | 44–59 |
| Baby Boomers (1946–1964) | 60–78 |
| Silent (1928–1945) | 79–95 |

A concrete age is drawn uniformly within the range for each persona, with one
plausibility constraint: the age is made consistent with the sampled education
(`_EDU_MIN_AGE` in `personas.py`), so a persona cannot hold a qualification it
is too young to have reached (degree from 21, other higher education from 19,
A-level from 18). This removes implausible combinations such as a 17-year-old
"with a degree" WITHOUT changing any sampled marginal: generation, education,
and the rest are untouched; only the otherwise-free within-band age draw is
conditioned on education. Other attribute pairs (e.g. income and education) are
still drawn independently.

---

## 4. Political affiliation (psychographic axis: SECOND source)

Used in [`personas_sim/config.py`](personas_sim/config.py) → `POLITICS`, which
powers the `psychographic` method.

- **Source:** 2024 UK General Election GB vote shares (House of Commons Library,
  *General Election 2024 results*). https://commonslibrary.parliament.uk/
- The `POLITICS` marginal is **derived**: GE2024 vote shares scaled to leave a
  `"no firm party allegiance"` bucket for the politically disengaged.
  Approximate by construction: a modelling input, not a verbatim statistic.
- **Honest limitation: this breaks the single-source property.** Everything
  else here comes from Yale CCBM 2024; political affiliation does not (it isn't
  in that report). So the opinion frame and the *demographic* frame still match
  by construction, but the *psychographic* axis is stitched in from a second
  publication. That's a deliberate, flagged trade-off to add the strongest UK
  climate-attitude driver the Yale demographics lack.
- The `psychographic` method runs on **Q1 only** (a cheap creative extension);
  the headline 5-method results are unaffected.

The climate-stance party grouping used by the political-gradient diagnostic
(`diagnostics.py`): **progressive/concerned** = Labour, Liberal Democrat,
Green, SNP/Plaid/other; **skeptic** = Conservative, Reform UK; `"no firm party
allegiance"` is excluded. This is a climate-concern proxy, not a strict
left–right axis (e.g. the Lib Dems are centrist but climate-progressive).

---

## 5. Psychological axes: openness and values (further inputs)

Two more psychographic axes (`config.py` → `OPENNESS`, `VALUES`) powering the
`openness` and `values` methods. Like `POLITICS`, these are **modelling
inputs**, not single-source Yale numbers. The rigour is in the gradient's
**known expected direction**, not in the marginal.

- **`OPENNESS`** (Big Five). Banded into tertiles (high / average / low), since
  the trait is roughly normally distributed; the marginal is an approximation,
  not a measured UK statistic. Expected gradient: higher openness → more
  climate concern (Soutter et al. 2020; Hirsh 2010).
- **`VALUES`** (Schwartz higher-order values). An illustrative marginal in the
  style of European Social Survey value priorities (self-transcendence tends to
  rank highest, self-enhancement lowest). Expected gradient: self-transcendence
  → more concern, self-enhancement → less (Schwartz 1992; Stern 2000).

Gradient groupings (`diagnostics.py`): openness high = `high openness`, low =
`low openness` (average excluded); values high = `self-transcendence`, low =
`self-enhancement` (`conservation` and `openness to change` excluded as
ambiguous for climate). The persona clauses deliberately avoid any
environment/climate wording, so a concern shift is a genuine downstream effect
of the trait, not a restatement of the outcome.

**Why not MBTI.** I considered a Myers-Briggs axis and rejected it: MBTI has
poor test-retest reliability and the dichotomous types lack empirical support,
and there is no established MBTI → climate-attitude relationship. Without an
evidence-based expected direction, its paired gradient would have no sign to
validate against, which is the bar every other axis here meets. Big Five
openness and Schwartz values are the psychometrically grounded, directional
alternatives.

---

## 6. AI attitudes (generality test, THIRD source)

Used in [`personas_sim/config.py`](personas_sim/config.py) → the `ai_llm_benefit`
question (topic="ai"). This is a deliberate test of whether the methods and the
psychographic-axis approach transfer beyond climate.

- **Source:** Ada Lovelace Institute & The Alan Turing Institute, *How do people
  feel about AI? Wave 2* (March 2025). Nationally representative UK survey,
  n=3,513, fieldwork November 2024.
  https://www.turing.ac.uk/news/publications/how-do-people-feel-about-ai
- **Item:** Figure 3, "To what extent do you think that the use of this
  technology will be beneficial?", **Large language models (e.g. ChatGPT)** row.
  Reported: Very 17%, Fairly 46%, Not very 13%, Not at all 6%. The trailing
  "Don't know / Prefer not to say" = 18% by subtraction (the four named bars sum
  to 82). Mapped to A=Very … D=Not at all, E=Don't know (sums to 1.00).
- **Honest notes:** a THIRD source (further from the single-publication ideal);
  frame is UK adults ~18+ (minor mismatch with the personas' 16+ frame, which
  comes from the Yale sample). The survey deliberately oversampled lower-income,
  lower-digital-skill, and Black/Asian respondents and then weighted, so the
  headline figures are nationally representative.
- **Why this item:** LLMs is the most general-purpose, on-brand AI use and has a
  good spread across categories (unlike, say, facial recognition at 91%
  beneficial, which would be near-degenerate).
- **Axis expectation for AI** (`AXES_BY_QID` in run.py): **Openness expected
  positive** (openness = openness to new technology); **political and values
  exploratory** (no established AI direction). This is how the "dominant axis is
  topic-dependent" hypothesis is encoded and tested.

---
---

# Part 2: References, the literature behind the design choices

## Foundations: personas reproducing survey distributions

- **Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D.
  (2023). *Out of One, Many: Using Language Models to Simulate Human Samples.* Political
  Analysis, 31(3), 337–351.** https://arxiv.org/abs/2209.06899
  - The "silicon sampling" / "algorithmic fidelity" paper: conditioning an LLM on
    demographic backstories recovers the opinion distributions of real subpopulations.
  - backs: the whole premise, and the `demographic` method
    (`personas.py:_make_prompt` / `PROMPT_ATTRIBUTES`).

- **Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023).
  *Whose Opinions Do Language Models Reflect?* (OpinionQA). ICML 2023.**
  https://arxiv.org/abs/2303.17548
  - Establishes the methodology of scoring LM opinion *distributions* against real
    survey data (Pew) by demographic group, and surfacing groups poorly represented.
  - backs: the TVD-vs-ground-truth evaluation (`evaluate.py`) and the age-gradient
    sub-group check (`diagnostics.py:age_gradient`).

---

## Methods

### `elicited`: verbalized sampling (soft per-persona distributions)

- **Zhang, et al. (2025). *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock
  LLM Diversity.* arXiv 2510.01171.** https://arxiv.org/abs/2510.01171
  - Training-free prompting strategy: ask the model to verbalize a probability
    distribution over responses to counter post-training mode collapse; lists social
    simulation as a target use case. This is the most direct citation in the repo.
  - backs: `personas.py:elicitation_prompt` and the `elicited` method in `run.py`.

- **Lin, S., Hilton, J., & Evans, O. (2022). *Teaching Models to Express Their
  Uncertainty in Words.* TMLR / arXiv 2205.14334.** https://arxiv.org/abs/2205.14334
  - Shows an LLM can emit calibrated "verbalized" probabilities without access to
    logits, and that they can also be miscalibrated.
  - backs: using self-reported probabilities as a stand-in for token logprobs (Claude's
    API doesn't expose them), and the stated miscalibration caveat.

### `network`: multi-agent discussion (social influence)

- **Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S.
  (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023.**
  https://arxiv.org/abs/2304.03442
  - Canonical demonstration of LLM agents forming opinions and influencing each other
    through conversation.
  - backs: the two-round discussion design (`personas.py:opinion_prompt`,
    `question_block_after_discussion`, and `run.py:_run_network`).

### Psychographic axes: demographics + an attitudinal driver

The premise that *attitudinal/identity* variables carry opinion signal the
standard demographic backstory misses. Three axes are implemented, each added
as one sentence and validated by a paired gradient
(`diagnostics.py:axis_gradient`) against `demographic` as a control.

- **Political identity** (`psychographic` method). Among the strongest
  correlates of UK climate attitudes in published polling, yet absent from the
  Yale demographic table, which is why it's the highest-leverage axis.
  - backs: `personas.py:_POLITICS_PHRASE`, `config.py:POLITICS`.

- **Big Five Openness** (`openness` method). Openness is the personality trait
  most consistently linked to environmental concern and pro-environmental
  behaviour.
  - **Soutter, A. R. B., Bates, T. C., & Mõttus, R. (2020). *Big Five and
    HEXACO Personality Traits, Proenvironmental Attitudes, and Behaviors: A
    Meta-Analysis.* Perspectives on Psychological Science, 15(4), 913–941.**
    https://doi.org/10.1177/1745691620903019
  - **Hirsh, J. B. (2010). *Personality and environmental concern.* Journal of
    Environmental Psychology, 30(2), 245–248.**
  - backs: `personas.py:_OPENNESS_PHRASE`, `config.py:OPENNESS`.

- **Schwartz values** (`values` method). Self-transcendence (universalism /
  benevolence) is the value orientation most strongly predictive of
  pro-environmental attitudes; self-enhancement (power / achievement) the least.
  - **Schwartz, S. H. (1992). *Universals in the content and structure of
    values.* Advances in Experimental Social Psychology, 25, 1–65.**
  - **Stern, P. C. (2000). *Toward a coherent theory of environmentally
    significant behavior* (value-belief-norm theory). Journal of Social Issues,
    56(3), 407–424.**
  - backs: `personas.py:_VALUES_PHRASE`, `config.py:VALUES`.

- **Why not MBTI.** Poor test-retest reliability, unsupported dichotomous
  types, and no established link to climate attitudes, so its paired gradient
  would have no expected sign to validate against. Rejected in favour of the
  evidence-based axes above. (See Part 1 §5.)

- **The dominant axis is topic-dependent (generality test).** The expected
  direction of each axis is *per question*, not global. For the AI question,
  Openness is the pre-registered expected-positive axis because openness to
  experience predicts technology acceptance and positive attitudes to new
  technology (a robust finding in the technology-adoption / Big Five
  literature), whereas political identity, which dominates climate attitudes,
  has no comparably established link to AI optimism and is therefore treated as
  exploratory. This is encoded in `AXES_BY_QID` (run.py) and is the basis of the
  "which axis to condition on depends on the question" finding.

### Anti-sycophancy framing

- **Sharma, M., Tong, M., Korbak, T., et al. (2023). *Towards Understanding Sycophancy in
  Language Models.* ICLR 2024 (Anthropic).** https://arxiv.org/abs/2310.13548
  - Documents that RLHF models default to the agreeable / socially-approved answer, the
    failure mode the "no right answers, don't hedge" instruction is built to counter.
  - backs: `personas.py:_ANTISYC`.

---

## Critiques & ceilings

- **Domínguez-Olmedo, R., Hardt, M., & Mendler-Dünner, C. (2024). *Questioning the
  Survey Responses of Large Language Models.* NeurIPS 2024.**
  https://arxiv.org/abs/2306.07951
  - LLM survey responses are governed by ordering/labeling biases (e.g. toward the
    letter "A"); correcting for them, models trend toward uniform-random answers. A core
    caveat for this whole approach.
  - backs: the `parse_letter` design (`llm.py`) and the "right for the wrong reason"
    diagnostics (peak / entropy / persona-dispersion).

- **Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R.,
  Liang, P., & Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People.*
  arXiv 2411.10109.** https://arxiv.org/abs/2411.10109
  - Interview-grounded agents replicate participants' General Social Survey answers ~85%
    as accurately as the participants replicate themselves two weeks later. This is the
    **primary source for the human self-replication ceiling** (~91% on these survey
    items) that bounds achievable distribution accuracy: survey respondents change their
    own answer ~19% of the time on re-asking, so even a perfect simulator caps there.
  - backs: `run.py:HUMAN_CEILING`.

- **Prompt-based synthetic-persona plateau.** Independent prior work on
  predicting survey-response distributions with prompted LLM personas reports
  the same ceiling this study reproduces: prompt-engineering tops out well short
  of human fidelity. See Suh et al. 2025 (SubPOP, below), who quantify the gap
  and turn to fine-tuning. The "distribution accuracy" metric used here is just
  `1 - TVD`, the standard OpinionQA-style scoring of LM opinion distributions
  against real survey data (Santurkar et al. 2023).

---

## Beyond prompting: the fine-tuning next step

- **Suh, J., Jahanparast, E., Moon, S., Kang, M., & Chang, S. (2025). *Language Model
  Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions*
  (SubPOP). ACL 2025.** https://arxiv.org/abs/2502.16761
  - Same problem framing: predicting survey response *distributions* across
    subpopulations. Finds that prompt-engineering struggles to do this faithfully, and
    instead fine-tunes on a 70K subpopulation-response dataset (SubPOP), cutting the
    LLM–human gap by up to 46%.
  - context: explains why the prompt-only methods here plateau (the framing Artificial
    Societies reports), and is the concrete fine-tuning direction beyond this repo's
    limitations (below).

---

## Metrics

TVD, Jensen–Shannon divergence, and Shannon entropy (`evaluate.py`) are standard
information-theoretic quantities and need no method paper. For JSD specifically see
**Lin, J. (1991). *Divergence measures based on the Shannon entropy.* IEEE Transactions
on Information Theory, 37(1), 145–151.**

---
---

# Honest limitations

- Attributes are mostly sampled **independently** from their marginals (the
  report gives marginals, not the joint). One dependency is enforced as a
  plausibility constraint: age is made consistent with education (see
  "Generation → age"), which removes the worst implausible combinations while
  preserving every marginal exactly. Other correlations (e.g. income with
  education) are still not modelled; full joint sampling from real cross-tabs is
  the next step.
- Ground-truth percentages are the report's rounded figures; use Appendix I
  for exact values if reporting to more than whole-percent precision.
- The generation→age ranges are a modelling choice (uniform within band);
  the real within-generation age distribution is not perfectly uniform.
