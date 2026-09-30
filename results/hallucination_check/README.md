# Hallucination check (2026-09-30)

A screening, not a replacement for the official scores. Official BLEnD scoring stays the primary analysis.

## 1. How often do the models abstain?
- 10 of 3,700 evaluable answers are a hedge or refusal ("there is no official data", "Iran has no Independence Day").
- Both models almost always give a concrete answer, even to questions where some BLEnD annotators said "I don't know".

## 2. Answers that match no BLEnD reference
- 1,257 of 3,700 answers (34%) match no reference.
- These are hallucination *candidates*, not hallucinations. A random sample of 96 was checked by hand: 12 per model × language × country.
- Unclear names were checked with a web search.
- The labels are an AI-assisted screening (Claude). Categories C and D should be confirmed by native-speaker reviewers.

| Category | In sample (n=96) | Estimated share of all 3,700 answers |
|---|---|---|
| A: correct, but missed by the scorer (spelling, Persian digits, synonyms) | 26 | ≈ 9% |
| B: plausible or real answer, not among the references | 60 | ≈ 21% |
| C: real entity, but clearly wrong | 7 | ≈ 3% |
| D: likely fabricated entity | 3 | ≈ 1% (95% CI 0–2%) |

- C and D together: ≈ 3.6% of all answers (95% CI 1.7–5.7%). Estimates are weighted by the number of unmatched answers in each cell.
- D (all in Azerbaijani answers):
  - "Oksigen" and "Qafqazın qartalı" given as sports films; no such films were found.
  - "Çiklatçı" given as a chocolate brand; no such brand was found.
- C examples:
  - Naseem Hamed (a British boxer) given as a famous Iranian boxer;
  - Turkish chefs given for Azerbaijan;
  - Nowruz called a religious holiday;
  - wrestling given as a track-and-field sport;
  - "Iran" given as Iran's own soccer rival;
  - Baku Crystal Hall given as a wedding venue;
  - the Baku Jazz Festival given as the biggest festival.
- Human reviewers (local-language validation sample): every error they confirmed involves a real entity, for example Veysəloğlu, kutab, kotlet and Saadabad Palace. None was fabricated.

## 3. Scoring artifact found along the way: Persian digits and ZWNJ
The Persian BLEnD prompt says «فقط اعداد را ارائه دهید» ("give numbers only"). Unlike the English prompt, it does not ask for Arabic numerals. The official scorer does not normalise:
- Persian digits (۱۰ vs 10);
- the zero-width non-joiner (چهارشنبه‌سوری vs چهارشنبه سوری).

Re-scoring the Persian answers with these normalised (English answers unchanged):

| Model | Persian, official | Persian, normalised | Gap (EN − FA), official | Gap (EN − FA), normalised |
|---|---|---|---|---|
| GPT-4.1 | 73.7 | 82.5 (+40 items) | −3.9 (p = .127) | −12.7 (p < .001) |
| Gemini 3.5 Flash-Lite | 65.1 | 77.9 (+58 items) | −7.5 (p = .015) | −20.2 (p < .001) |

- No item flips from correct to incorrect.
- 95 of the 98 new matches are genuine. The other 3 are substring matches, the same rule the official scorer applies to Latin digits.
- The Azerbaijani and English answers are not affected.
- This explains much of the under-counting the Iran reviewers found (8 of 11 automatic "incorrect" judged correct).
- Consequence: the official score under-estimates Persian accuracy, so the Persian advantage in Iran is larger than reported.

Files:
- `unmatched_sample_labelled.csv`: the 96 checked answers with category and note;
- `persian_normalisation_flips.csv`: the 98 Persian answers that become correct after normalisation.
