# LLM Persona Simulation: do synthetic populations reproduce real UK opinion?

## Problem

Can a population of LLM "personas" reproduce the *distribution* of opinions a
real human group holds, not just give a plausible-sounding answer? This
matters because a simulated population is only useful as market research if
it is **statistically faithful** to the real one, not merely realistic on
the surface.

I frame it as a measurable question: build personas, ask them several
single-select survey questions, and measure the distance between their answer
distributions and **real published survey results**.

## Results (headline)

100 personas, 5 methods, 3 Yale CCBM 2024 climate-attitude questions, run on
**two models**: `claude-haiku-4-5` and `claude-sonnet-4-6`. Distribution
accuracy = `1 − TVD`; human-replication ceiling ≈ 91%.

| method | Haiku avg | Sonnet avg | what it tells us |
|--------|:---:|:---:|---|
| baseline | 40.0% | 32.7% | no-persona model is degenerate: refuses/hedges, collapses to one option |
| demographic | 63.3% | 65.3% | the synthetic-persona plateau (~60–67%, as prior work reports), stable across models |
| rich | 54.7% | 63.0% | elaboration *hurts* on Haiku, *helps* on Sonnet |
| network | **45.7%** | **67.0%** | **flips from worst to best** (see finding 1) |
| elicited | 93.8% | 85.2% | highest on both, but the *drop* is the diagnostics working (see finding 3) |

**Three findings, and the comparison is the finding:**

1. **Mode collapse is largely a model-capability artifact, and `network`
   inverts.** On Haiku the standard methods mode-collapse onto the central
   answer (peaks 76–92% vs truth 30–45%) and *elaboration makes it worse*:
   peer discussion (`network`) is the **worst** method (45.7%) via conformity
   collapse. On Sonnet the peaks fall toward truth and `network` becomes the
   **best** non-elicited method (67.0%): stronger agents in discussion produce
   *diversity* instead of converging. A method's ranking can invert between
   models, so "which persona method is best" is the wrong question without
   naming the model.
2. **Real demographic structure emerges only on the stronger model.** The age
   gradient (younger people are more concerned) is **absent or inverted on
   Haiku** (e.g. `demographic` −0.21, wrong sign) but **clearly positive across
   all four methods on Sonnet** (+0.17 to +0.53). Hitting the marginal and
   capturing the sub-group structure are different bars, and Sonnet clears the
   second where Haiku can't.
3. **`elicited` is highest on both, but higher isn't better, and the
   diagnostics prove it.** On Haiku it *beats* the 91% ceiling (93.8%) with a
   flat age gradient and low persona-dispersion (0.18), so the model is
   **reciting a well-calibrated population prior**, not simulating distinct
   people (which wouldn't transfer to a question with no published answer). On
   Sonnet it drops *below* the ceiling (85.2%), dispersion rises (0.21), and the
   gradient turns clearly positive (+0.26): **more genuine per-persona
   simulation.** Haiku's bigger number was the less trustworthy one.

The through-line: **getting the right distribution and getting it for the right
reasons are different problems**, model capability drives both, and only
purpose-built diagnostics (peak, age-gradient, persona-dispersion) tell them
apart. Full numbers print to stdout and `results.json`; see
[NOTES.md](NOTES.md) for the reasoning behind every choice. *(N=100, no
bootstrap CI yet; see limitations.)*

### Psychographic axes: one sentence breaks the plateau (Sonnet, Q1)

The 5 methods above all condition on *demographics only*. Each axis method adds
**one extra sentence** for an attitudinal driver the demographics miss;
`composite` adds all three at once. Nothing else changes. On
`claude-sonnet-4-6`, Q1 (ground-truth peak 30%):

| method | accuracy | peak | gradient (in-prompt vs control) |
|--------|:---:|:---:|:---:|
| demographic (control) | 72.5% | 55%, collapsed onto "Very important" | n/a |
| **political** | **91.5%** | 32%, near truth | **+1.77** vs +0.08 |
| values | 78.5% | 37% | **+1.65** vs −0.09 |
| openness | 77.5% | 41% | **+1.30** vs −0.08 |
| composite (all 3) | 80.5% | 37% | see below |

Three findings, and the third is the surprise:

1. **All three axes induce genuine, validated sub-group structure.** Every
   in-prompt gradient is strongly positive while every demographic control is
   flat (same personas, same run, the one extra sentence the only difference).
   So the political win is not a fluke of politics: adding *any* evidence-based
   attitudinal axis reliably makes the personas individuate in the right
   direction, and the paired control proves it each time. The political axis
   alone lifts accuracy +19 points and roughly halves mode collapse (peak 55% →
   32%, near the true 30%), nudging the 91% ceiling.
2. **A strong gradient does not guarantee the best marginal.** Values has almost
   the same gradient as politics (+1.65 vs +1.77) but a far smaller accuracy
   gain (+6 vs +19). The gradient measures whether the axis moves concern in the
   right *direction*; the marginal accuracy depends on how well that trait's
   *population distribution* maps onto this question's real answer distribution.
   Party identity maps closest to UK climate opinion, so it wins on accuracy.
   Having both diagnostics is what separates "is the axis real?" from "does it
   improve the aggregate?"
3. **Stacking does NOT compound; the kitchen sink is worse than the best single
   axis.** `composite` (80.5%) lands *below* `political` (91.5%) and near the
   weaker axes: blending in the differently-shaped openness/values signals
   dilutes the clean political one (composite inherits values' over-spreading
   into D = 0.25 vs truth 0.13). And the composite's gradients survive
   **unequally** (political +1.33, openness +1.13, **values +0.06, collapsed**
   from +1.65): when axes compete in one prompt, the model keeps the cues it
   weights most and effectively drops the weakest. So combining axes doesn't
   just dilute accuracy, it changes *which* structure the persona encodes. The
   practical lesson: find the single axis that best maps to your target outcome
   and use it; don't pile on.

This sharpens the headline: the synthetic-persona plateau isn't a hard ceiling;
it's partly an artifact of conditioning on **demographics only**. Add *the right*
validated psychographic axis those demographics miss and the population
de-collapses with real sub-group structure, unlike `elicited`, which hits the
marginal by reciting it. *(Caveat: Q1 only, single run with ~3pt of LLM sampling
noise, and the openness/values marginals are approximate modelling inputs; see
[SOURCES.md](SOURCES.md) and limitations.)*

### Why this is a market-research problem

Consumer-sustainability and ESG attitudes are a core market-research category
(brand-sustainability scoring, willingness to pay a green premium, ESG-fund
demand, climate-related reputational risk). Yale's *Climate Change in the
British Mind 2024* is the gold-standard UK distribution for these attitudes,
which makes it a clean ground-truth target for a "can we simulate this
audience?" eval. If a method can hit the published distribution on three
related climate-attitude items, that's directly relevant to how a vendor
would simulate, e.g., a UK consumer panel for an ESG-positioning study,
without needing to recruit one.

## Ground truth

Yale Program on Climate Change Communication, *Climate Change in the British
Mind, 2024* (Leiserowitz et al., 2025). Nationally representative survey of
UK residents aged 16+, 10,660 interviews, 7–13 Nov 2024, MoE ±0.9pts.

Three single-select items are used (see [config.py](personas_sim/config.py)):

1. **personal_importance** *(Q2.3)*: "How important is the issue of climate
   change to you personally?"  Five-point scale.
2. **worry** *(Q2.1)*: "How worried are you about climate change?"
   Four-point scale.
3. **harm_personally** *(Q2.2)*: "How much do you think climate change
   will harm you personally?"  Five-point scale (incl. "Don't know").

Each question's ground truth is taken from the published CCBM 2024 report;
the `source` field on every question records the provenance and prints at
runtime.

Three items is enough to ask: **does a method's accuracy generalise across
questions, or did it get lucky on one?** Per-question accuracies are reported
alongside the average. The three questions also span both 4-point and
5-point scales, so the pipeline is exercised across different option
counts.

## Method: five approaches, compared

The comparison *is* the finding, so I run five methods against the same
target on every question. They're points on a spectrum from "no conditioning"
to "fix the things that actually make synthetic personas fail."

1. **baseline**: no persona at all (null model).
2. **demographic** *(silicon sampling, Argyle et al. 2023)*: 100 personas
   sampled to match UK 16+ demographics (generation, sex, region, education,
   income, from the survey's own weighted sample composition), with a short
   factual prompt.
3. **rich**: same sampled attributes, fuller first-person backstory prompt.
4. **network** *(beyond synthetic-persona; generative agents, Park et al. 2023)*: a **toy multi-agent discussion**.
   Personas are split into discussion groups of 5. **Round 1:** each persona
   writes ONE short opinion sentence in their own voice (free text, no letter
   yet). **Round 2:** each persona is shown the other four members' opinions
   and then picks a multiple-choice letter. They may stick or shift after
   hearing the group. The round-2 letters are the method's output. This targets
   the AS thesis that **opinions form and propagate socially**, not in
   isolation.
5. **elicited** *(verbalized sampling, Zhang et al. 2025)*: instead of forcing ONE hard letter,
   each persona reports a **probability over every option**; we average the
   soft per-persona distributions. Same personas as `rich`, so the elicitation
   *format* is the only variable. This attacks the dominant synthetic-persona
   failure head-on: averaging hard votes throws away within-person uncertainty
   and bakes in **mode collapse** (a 55/45 persona always shows as 100% A).
   A soft distribution keeps that uncertainty, so the population spread is
   recovered rather than flattened. *(With a provider that exposes token
   logprobs you could read the per-option distribution directly; Claude's API
   doesn't, so verbalized sampling is the equivalent.)*

Methods 1–3 are the standard "synthetic persona" approach that prior work
(e.g. the Artificial Societies Jan 2026 eval report) shows plateauing around
60–67% distribution accuracy across 1,000 surveys, a limit also reported for
prompt-engineering by Suh et al. 2025 (SubPOP), who turn to fine-tuning instead.
Methods 4 and 5 are first-principles attempts at the two things that report says
are missing: social dynamics (network) and honest within-person uncertainty
(elicited).

**Psychographic axes *(Q1-only extensions)*.** Every method above conditions on
*demographics* only, the textbook silicon-sampling recipe. The extension adds
three methods that each take the `demographic` prompt and append **one extra
sentence** for an attitudinal driver the demographics miss:

- `psychographic`: UK political identity (party lean, from 2024 GE vote shares).
- `openness`: Big Five Openness level (intellectually curious vs practical).
- `values`: Schwartz dominant value (self-transcendence vs self-enhancement).

The extra sentence is the *only* thing that differs from `demographic`, so each
is a clean test of whether that axis adds signal. Each is validated by a
**paired gradient with a known expected direction** (see Evaluation), and each
runs on Q1 only as a cheap add-on in its own block, so the headline 5-method
results are untouched. They share a small `AXES` registry, so adding a fourth
axis is a few lines.

A `composite` method then **stacks all three axes** on one persona (demographic
base + all three sentences) to ask whether the per-axis gains *stack or
saturate*, and whether all three sub-group gradients survive together or
interfere. (They survive but attenuate: splitting a composite persona on one
axis leaves the other two as within-group noise, diluting each gradient.) *(I considered a personality typology like MBTI but
rejected it: it has poor reliability and no established link to climate
attitudes, so its gradient would have no expected sign to validate against.
Openness and Schwartz values are evidence-based and directional. This also
introduces extra data sources beyond Yale; see [SOURCES.md](SOURCES.md).)*

**Anti-sycophancy framing.** Every persona prompt also carries an explicit
"there are no right answers, many people genuinely disagree, don't hedge"
instruction. LLMs default to the cautious central answer (Sharma et al. 2023),
which is itself a driver of mode collapse; licensing disagreement is a cheap,
first-principles counter applied across all persona methods.

Personas are sampled from the **survey's own weighted sample composition**
(Yale CCBM 2024, Appendix III), so the demographic frame and the opinion frame
come from one source and match by construction (UK 16+), no ONS stitching
required. See [SOURCES.md](SOURCES.md) for provenance and the modelling
choices (which attributes are used, how the "prefer not to say" buckets are
handled, the generation→age mapping). The same persona pool is reused across
all questions so method differences aren't contaminated by demographic
sampling noise.

## Evaluation

The headline metrics are matched to the framing in Artificial Societies'
Jan 2026 eval report; the rest exist to catch *being right for the wrong
reason*.

- **Distribution accuracy** = `1 − TVD`, as a percentage: the share of
  response mass correctly allocated across the question's options. Reported
  per question and averaged across questions. Also reported: **JSD**.
- **Response consistency**: for a sample of personas, ask the same question
  in 5 different wordings; report mean within-persona modal-agreement. Run on
  Q1 only (consistency is a property of persona stability, not the question).
  A faithful persona should be stable to paraphrase. Computed for the
  hard-vote methods.

**Why accuracy alone isn't enough: two diagnostics.** A high marginal can be
a coincidence: a mode-collapsed population (everyone says the same thing) can
still average to roughly the right distribution. So I report:

- **Peak (modal-bucket share)** next to the ground-truth peak. A method whose
  peak sits far above truth's has funnelled the population onto one option.
  This is exactly how mode collapse shows up, and it's the clearest single
  signal of why a method fails. (Entropy is in `results.json` as the same
  signal from the spread side.)
- **Age-gradient sub-group check** *(automated)*: in the real UK data younger
  people are more concerned about climate. The harness splits personas into
  young (<40) and old (≥60), scores each on a concern scale, and reports the
  gradient. If the marginal is right but the gradient is flat or wrong-signed,
  the conditioning isn't doing real work; the marginal was luck. This is the
  test that separates a genuine result from a lucky average, and it doubles as
  a check that the demographic conditioning actually bites.
- **Elicited persona-dispersion** *(automated)*: mean pairwise TVD between
  the `elicited` method's per-persona distributions. If it's ~0 the personas
  are near-identical, which means a near-perfect averaged marginal is the model
  **reciting an aggregate it already knows**, not simulating distinct people,
  and recitation can't generalise to a question with no published answer. This
  is what stops a ceiling-beating `elicited` score from being mistaken for
  fidelity.
- **Paired axis-gradient checks** *(automated; one per psychographic axis)*: for
  each axis there is a known real-world direction (climate-progressive voters,
  higher-openness people, and self-transcendent people are all more concerned).
  The harness computes the gradient with the axis **in** the prompt (expect
  clearly positive) **and** for `demographic` as a **control** with the axis
  **out** of the prompt (expect ≈flat). Positive where conditioned, flat where
  not = proof the axis is actually driving the answers, not leaking in via
  correlated demographics. The known expected sign is what makes this a real
  test, and it's exactly why MBTI was rejected (no expected sign to check).

**Human-replication ceiling.** Real respondents change their own answer
~19% of the time on re-asking (Stanford), so even a perfect simulator caps
around **91%** distribution accuracy. That's the bar, not 100%.

- **Unparseable rate**: share of replies that didn't yield a valid answer
  (data-quality check; reported rather than silently dropped).

## Run it

```bash
# 1. Create + activate the conda env (Python 3.11)
conda create -n llm-persona python=3.11 -y
conda activate llm-persona

# 2. Install deps
pip install -r requirements.txt

# 3. Offline smoke-test (no API key)
USE_MOCK=1 python -m personas_sim.run

# 4. Real run (full 5-method suite, 3 questions + Q1 psychographic extension)
export ANTHROPIC_API_KEY=sk-ant-...
USE_MOCK=0 python -m personas_sim.run

# 5. Quick creative add-on only: the psychographic axes vs demographic on Q1
USE_MOCK=0 N=100 EXT_ONLY=1 python -m personas_sim.run
# ...or one axis at a time (cheaper): AXIS in {political, openness, values}
USE_MOCK=0 N=100 EXT_ONLY=1 AXIS=openness python -m personas_sim.run
```

Outputs: per-question + averaged tables (stdout), `results.json`, and
`comparison.png` (one subplot per question + an averaged-accuracy summary).
`EXT_ONLY=1` writes a separate `results_psychographic.json` and leaves the main
results untouched.

**Cost / runtime.** Per question the full run makes 100 baseline + 100
demographic + 100 rich + 200 network + 100 elicited = 600 calls. Across 3
questions that's ~1,800 calls, + ~150 consistency + 400 for the Q1 psychographic
axes (3 axes + composite), about **2,350 calls total**. With `claude-haiku-4-5`
(set in [llm.py](personas_sim/llm.py)) expect roughly 20–40 minutes sequential;
override with `MODEL=claude-sonnet-4-6` for higher fidelity, or `N=20` for a
quick look. The `EXT_ONLY=1` fast path is ~500 calls at N=100 (demographic + 3
axes + composite); `AXIS=` runs just one axis (skips composite).

## Structure

```
personas_sim/
  config.py        # questions, ground truths, demographics + POLITICS (Yale + GE2024)
  llm.py           # ask_persona() + parse_letter() + parse_distribution()
  personas.py      # build personas (demographic/rich/network/elicited + axes: psychographic/openness/values)
  evaluate.py      # accuracy / JSD / peak / entropy; hard-vote & soft-dist paths
  consistency.py   # within-persona stability across question rephrasings
  diagnostics.py   # age-gradient, persona-dispersion, political-gradient checks
  run.py           # 5-method runner + Q1 psychographic extension, reporting, plot
tests/
  test_evaluate.py # metric + parsing + diagnostics unit tests (offline)
```

Run the tests (no API key needed):

```bash
python tests/test_evaluate.py        # or: python -m pytest tests/ -q
```

## Related work

The methods here are named implementations of specific results; full citations
and the choice-by-choice mapping are in [SOURCES.md](SOURCES.md).

- **Silicon sampling**: conditioning an LLM on demographics to recover
  subpopulation distributions (Argyle et al. 2023); scoring LM opinion
  *distributions* against real survey data (Santurkar et al. 2023, OpinionQA).
- **`elicited` / verbalized sampling**: prompting for a probability
  distribution to counter mode collapse (Zhang et al. 2025); verbalized
  uncertainty and its calibration caveat (Lin et al. 2022).
- **`network`**: LLM agents forming and propagating opinions through
  conversation (Park et al. 2023, *Generative Agents*).
- **Anti-sycophancy**: the RLHF agreeable/cautious default it counters
  (Sharma et al. 2023).
- **Critiques & ceiling**: ordering/labeling bias and collapse-to-uniform in
  LLM survey responses (Domínguez-Olmedo et al. 2024); the ~85–91% human
  self-replication ceiling (Park et al. 2024, *1,000 People*).
- **Beyond prompting**: fine-tuning on scaled survey data as the alternative to
  the prompt-only plateau (Suh et al. 2025, *SubPOP*).

## Honest limitations / next steps

- Demographic attributes are sampled **independently** (marginal
  distributions), so age–education and age–income correlations aren't
  modelled and some persona combinations are less plausible than real
  respondents. Conditional/joint sampling is the obvious next step.
- The `network` method is the simplest possible model of social influence
  (one round of peer-opinion exposure in 5-person groups). Richer variants
  (variable group size, multiple rounds, persona-weighted influence, real
  network topology, multi-turn conversation rather than one-shot opinion)
  are the obvious next step.
- Consistency uses 5 hand-written rephrasings on 10 personas per method, on
  the first question only. Larger N, machine-generated paraphrases, and
  coverage across all questions would tighten the estimate.
- The `elicited` method trusts the model's *self-reported* probabilities,
  which can be miscalibrated. Reading token logprobs over the option letters
  (a provider that exposes them) would give a model-grounded distribution
  instead of a verbalized one.
- The age-gradient check uses a coarse young(<40)/old(≥60) split and an
  ordinal concern score; it's a directional sanity check, not a calibrated
  sub-group fit. Per-generation calibration against the report's own
  breakdowns would be stronger.
- No statistical test on the *difference between methods* yet: with 100
  personas there is sampling noise; a bootstrap CI on accuracy would tell
  us whether method differences are real or noise.
- Mock model results are illustrative only; real-model numbers will differ.
  In particular the mock's elicited distributions are near-identical across
  personas, which flatters that method offline.
- All five methods are **prompt-only**. The biggest lever beyond prompting is
  **fine-tuning on scaled survey data** (Suh et al. 2025, *SubPOP*, report a 46%
  reduction in the LLM–human gap), which is the natural next step past the
  prompt-engineering plateau these methods sit on.

## If time permitted: creative directions

Higher-effort ideas I'd reach for next, ordered by expected leverage. Each
ties to something the results already hint at.

1. **RAG-grounded personas.** Replace the
   model's *prior* about a group with *evidence*: retrieve a handful of real
   opinions (social-listening, forum, or comment data) matched to each persona's
   demographic and inject them into the prompt. This is the closest a prompt-only
   method can get to AS's 2M real-world profiles, and it attacks the failure the
   dispersion diagnostic exposed: personas reciting an aggregate instead of
   holding distinct, grounded views.
2. **A psychographic stack, not just politics** *(partly done)*. One political
   sentence moved Q1 accuracy +17 points, so I generalised the axis framework
   and added two more validated axes: **Big Five Openness** and **Schwartz
   values** (each with its own paired gradient). The natural next axes are media
   diet, religiosity, urban/rural, and concrete behavioural anchors ("hasn't
   flown in three years", "drives a diesel", "shops at Aldi"); behaviour
   constrains opinion more tightly than demographics. The `composite` method
   already *combines* all three axes in one persona; the open question is
   whether a richer stack keeps stacking or saturates, and how to disentangle
   the gradients once axes are combined.
3. **Joint sampling from real microdata.** Personas are currently drawn from
   independent marginals, which produces implausible combinations. With the raw
   CCBM respondent rows (available on request from YPCCC) I'd sample real
   demographic *combinations*, so the synthetic population is a true shadow of
   the real one, then condition on them. This is the cleanest fix for the
   independence limitation and would sharpen every sub-group check.
4. **The honest generalisation test: an unpublished question.** The sharpest
   result here is that `elicited` hits the marginal by *reciting* a known
   distribution. The decisive experiment is to ask a question with **no published
   UK answer**, collect a small real sample to score against, and see which
   methods hold up. Methods that earned their accuracy through genuine
   conditioning (`psychographic`) should survive; reciters should collapse. That
   turns "right for the right reasons" from a diagnostic into a held-out test.
5. **Within-persona coherence, not just per-question marginals.** Real
   respondents' answers correlate across items (someone who calls climate
   "extremely important" tends to be "very worried"). I'd have each persona
   answer the whole battery and check whether the *within-persona correlation
   structure* matches the real joint distribution, not just the three marginals.
   A population can match every marginal and still be individually incoherent;
   this catches that.
6. **Richer social dynamics.** `network` flipped from worst to best on the
   stronger model, so social influence is clearly doing real work. I'd push it
   further: multi-round discussion, a homophilous network graph instead of random
   5-person groups, influence weighted by persona, and a bounded-confidence
   opinion-dynamics model, to test whether structured social topology beats a
   single round.
7. **Statistical rigour.** Bootstrap confidence intervals on accuracy (resample
   personas) so method differences under a few points can be called real or
   noise, plus repeated runs to quantify the LLM sampling variance we already saw
   (~4 points between Haiku runs). Cheap, and it would let every claim above be
   stated with error bars.
