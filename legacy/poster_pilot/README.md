# BLEnD Poster Pilot

## Working title
**Beyond Accuracy: Cross-Lingual Consistency of Cultural Knowledge in Modern LLMs on the BLEnD Benchmark**

## What this pilot tests
For each BLEnD item, the same cultural question is presented in:
1. English
2. The BLEnD local language (Persian for Iran; Azerbaijani for Azerbaijan)

We measure:
- English accuracy
- Local-language accuracy
- paired success (both versions correct)
- language-dependent failure (English correct, local wrong)
- reverse language-dependent failure (local correct, English wrong)
- answer-pair consistency for a qualitative subset

## Important framing
This is a **small-scale replication + focused extension**, not a claim that cross-lingual consistency is a new research problem. Prior work has already studied cross-lingual factual consistency. The contribution here is a focused BLEnD case study of everyday *cultural* knowledge, contrasting English vs. local-language access and analyzing the errors that create the gap.

## Dataset
Official BLEnD repository:
https://github.com/nlee0212/BLEnD

Short-answer annotations:
- Iran: data/annotations/Iran_data.json
- Azerbaijan: data/annotations/Azerbaijan_data.json

Question files:
- Iran: data/questions/Iran_questions.csv
- Azerbaijan: data/questions/Azerbaijan_questions.csv

## Pilot sample
23 questions total:
- 12 Iran
- 11 Azerbaijan
- selected across Food, Education, Holidays/Celebration/Leisure, and Sport
- selected to avoid obvious no-answer/not-applicable items where possible

## Models
The script is configured for two OpenAI-compatible models. Example current OpenAI models can be set through environment variables. Do NOT write API keys into the script.

## Reproducibility
Set:
OPENAI_API_KEY
MODEL_A
MODEL_B

Then run:
python run_experiment.py

The script saves:
- raw_responses.csv
- summary_by_model_and_language.csv
- paired_results.csv

## Why we do not fabricate results
The actual LLM calls require an API key/account. The package therefore prepares and executes the experiment but does not invent model outputs.
