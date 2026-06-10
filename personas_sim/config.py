"""
Central config: the survey questions, the real-world ground-truth distributions,
and the demographic distribution we sample personas from.

GROUND TRUTH SOURCE
-------------------
Leiserowitz, A., Carman, J., Goddard, E., Verner, M., Rosenthal, S., & Marlon, J.
(2025). Climate Change in the British Mind, 2024. Yale University / Yale Program
on Climate Change Communication.
  - Nationally representative survey of UK residents aged 16+
  - 10,660 interviews, 7-13 November 2024, MoE +/-0.9pts @ 95%

All three questions use Yale CCBM 2024 figures as reported:
  Q1 ("personal importance") = Q2.3, Q2 ("worry") = Q2.1,
  Q3 ("harm personally") = Q2.2.
Each question's `source` field records its provenance and prints at runtime.
Percentages are the report's body-text / data-table figures; some columns sum
to 99-101 due to source rounding (harmless -- distributions are normalised at
evaluation time). See SOURCES.md for the full mapping.

DEMOGRAPHIC SOURCE
------------------
Yale CCBM 2024, Appendix III "Sample Demographics" -- the WEIGHTED percentages
(the weighting is what makes the sample nationally representative, so weighted
is the right column to sample from). Using the survey's own sample composition
means the demographic frame and the opinion frame come from ONE source and
match by construction (UK 16+), rather than stitching ONS tables onto a Yale
survey. See SOURCES.md.

The full table is preserved below for transparency, including the
"prefer not to say" / "don't know" buckets. Persona construction uses a
subset of attributes and drops those uninformative buckets (renormalising the
rest) -- see PROMPT_ATTRIBUTES in personas.py.
"""


#########################################################
# QUESTIONS
#########################################################
# Each question carries its own options letters, ground-truth distribution, and
# (optional) rephrasings used by the consistency check. Distribution accuracy
# is computed per question and averaged across questions in the report.

QUESTIONS = [
    {
        "id": "personal_importance",
        "text": "How important is the issue of climate change to you personally?",
        "options": {
            "A": "Extremely important",
            "B": "Very important",
            "C": "Somewhat important",
            "D": "Not too important",
            "E": "Not at all important",
        },
        # Sums to 0.99 (source rounds to whole percents: 22/29/30/13/5)
        "ground_truth": {"A": 0.22, "B": 0.29, "C": 0.30, "D": 0.13, "E": 0.05},
        "source": "Yale CCBM 2024, Q2.3 (verified against published report)",
        "rephrasings": [
            "How important is the issue of climate change to you personally?",
            "When you think about climate change, how big a personal issue is it for you?",
            "On a personal level, how important do you consider climate change to be?",
            "How much does the topic of climate change matter to you personally?",
            "Personally speaking, how important to you is the issue of climate change?",
        ],
    },
    {
        "id": "worry",
        "text": "How worried are you about climate change?",
        "options": {
            "A": "Very worried",
            "B": "Somewhat worried",
            "C": "Not very worried",
            "D": "Not at all worried",
        },
        "ground_truth": {"A": 0.35, "B": 0.45, "C": 0.15, "D": 0.05},
        "source": "Yale CCBM 2024, Q2.1 (verified against published report)",
    },
    {
        "id": "harm_personally",
        "text": "How much do you think climate change will harm you personally?",
        "options": {
            "A": "A great deal",
            "B": "A moderate amount",
            "C": "Only a little",
            "D": "Not at all",
            "E": "Don't know"
        },
        "ground_truth": {"A": 0.17, "B": 0.4, "C": 0.29, "D": 0.11, "E":0.04},
        "source": "Yale CCBM 2024, Q2.2 (verified against published report)",
    },
]


#########################################################
# DEMOGRAPHICS  (Yale CCBM 2024, Appendix III, weighted %)
#########################################################
#
# Faithful transcription of the survey's own weighted sample composition.
# Frame: UK 16+. Some columns sum to 99-101 due to source rounding -- this is
# expected and harmless (probabilities are renormalised at sample time).
# `generation` is used instead of age bands because that is how Yale report it;
# personas.py maps each generation to a representative 2024 age for the prompt.
#
# Not every attribute or bucket is used to build personas. See PROMPT_ATTRIBUTES
# in personas.py for which are sampled, and how "prefer not to say" /
# "don't know" buckets are dropped and renormalised.

DEMOGRAPHICS = {
    "sex": {
        "male":                     0.47,
        "female":                   0.52,
        "another gender identity":  0.02,   # sums to 101 (rounding); fine
    },
    "generation": {
        "Gen Z (1997 or later)":    0.16,
        "Millennials (1981-1996)":  0.27,
        "Gen X (1965-1980)":        0.26,
        "Baby Boomers (1946-1964)": 0.27,
        "Silent (1928-1945)":       0.05,
    },
    "education": {
        "qualifications level 1 or below":     0.05,
        "GCSE / O-level / NVQ2":               0.19,
        "A-level / Scottish Higher / NVQ3":    0.28,
        "other higher education below degree": 0.04,
        "degree level or above":               0.27,
        "another qualification":               0.06,
        "no qualification":                    0.04,
        "prefer not to say":                   0.07,
    },
    "income": {
        "<26K":              0.20,
        "26K-52K":           0.27,
        "52K-75K":           0.12,
        "75K-100K":          0.08,
        "100K-150K":         0.06,
        "150K+":             0.03,
        "dont know":         0.10,
        "prefer not to say": 0.14,
    },
    "region": {
        "England":          0.84,
        "Scotland":         0.08,
        "Wales":            0.05,
        "Northern Ireland": 0.03,
    },
    # Available in the source but deliberately NOT used in persona prompts
    # (adds noise without a strong link to climate opinion; see SOURCES.md).
    "ethnicity": {
        "White":                       0.84,
        "Mixed / Multiple":            0.03,
        "Asian":                       0.07,
        "Black / African / Caribbean": 0.02,
        "other ethnic group":          0.01,
        "dont know":                   0.00,
        "prefer not to say":           0.02,
    },
}


#########################################################
# POLITICS  (psychographic axis -- SECOND data source)
#########################################################

# Political affiliation is the one strong UK climate-attitude driver that is
# NOT in the Yale demographics. Adding it powers the `psychographic` method.

# HONEST NOTE: this introduces a SECOND real-world source, so the repo's
# "every number from one publication" property holds for the opinion and
# demographic frames but NOT for this axis. See SOURCES.md.

# Derived from 2024 UK General Election GB vote shares (House of Commons
# Library, GE2024 results), scaled to leave a "no firm party allegiance"
# bucket for the politically disengaged. Approximate by construction.
POLITICS = {
    "Labour":                    0.30,
    "Conservative":              0.21,
    "Reform UK":                 0.13,
    "Liberal Democrat":          0.11,
    "Green":                     0.06,
    "SNP / Plaid / other":       0.04,
    "no firm party allegiance":  0.15,
}
