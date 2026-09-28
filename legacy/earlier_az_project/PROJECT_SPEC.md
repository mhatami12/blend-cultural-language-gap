# BLEnD Azerbaijani Poster Study — Final Scope

## Working title
**Beyond Accuracy: Cross-Lingual Consistency of Cultural Knowledge in Azerbaijani on the BLEnD Benchmark**

## Core research question
Does the language in which the same Azerbaijani cultural question is asked affect an LLM's ability to retrieve the correct cultural knowledge?

## Research questions
**RQ1.** Is accuracy different when the same BLEnD cultural question is asked in English versus Azerbaijani?

**RQ2.** How often does correctness depend on the query language at the question level (English correct / Azerbaijani wrong, and the reverse)?

**RQ3.** What recurring failure patterns appear among the language-discordant cases?

## Hypotheses
**H1.** English accuracy will be higher than Azerbaijani accuracy for this low-resource language.

**H2.** A non-trivial subset of questions will be language-dependent: the model will be correct in one language and wrong in the other.

**H3.** The most common discordant cases will involve incomplete, generic, or culturally inappropriate answers rather than purely random failures.

## Dataset
Use ONLY the original BLEnD data under `data/` (not `data_SemEval/`). The current official repository keeps the original BLEnD files and adds the 2026 SemEval material separately under `data_SemEval/`.

Azerbaijan has 500 short-answer cultural questions, each with an English question and an Azerbaijani version, plus human annotations and vote counts.

## Sample
Select **300 of the 500 paired questions**, stratified by topic, with a fixed random seed (42). This gives a reproducible, balanced-enough sample while keeping API workload manageable.

## Prompt control
Use BLEnD's official `inst-4` prompt only. It gives a single direct-answer instruction in English and Azerbaijani. We do NOT use `pers-3` in the main experiment because its persona instruction can introduce an additional experimental factor.

## Models
Run two multilingual LLMs. The exact model IDs are set through environment variables so the study can be reproduced without changing code.

Recommended setup when available:
- MODEL_A: a current OpenAI model
- MODEL_B: a second current multilingual model

The project is still valid as a single-model case study if API availability makes the second model impractical; the paired language analysis remains the main contribution.

## Main metrics
1. **Accuracy** in English and Azerbaijani using BLEnD's official short-answer scoring.
2. **Paired outcome pattern** for every question:
   - both correct
   - English-only correct
   - Azerbaijani-only correct
   - both wrong
3. **McNemar's exact test** on paired correctness to test whether English and Azerbaijani correctness rates differ.
4. **Language gap** = English accuracy − Azerbaijani accuracy (percentage points).

## Qualitative analysis
Inspect only a small set of discordant pairs (target 15–20 total across models):
- English correct / Azerbaijani wrong
- Azerbaijani correct / English wrong

Use transparent error labels only when the evidence is clear:
- incomplete answer
- generic answer
- wrong cultural item/entity
- irrelevant answer
- response-language mismatch
- other/uncertain

Do not claim fine-grained cultural errors where the researcher is not confident. BLEnD human annotations are the gold standard.

## Contribution statement
This is **not** a claim that cross-lingual consistency is a new problem, nor that Azerbaijani has never been evaluated. The contribution is a focused BLEnD case study that moves beyond aggregate language scores to **question-level paired consistency and language-dependent failure analysis for Azerbaijani**.
