# Data license and attribution

## BLEnD (third-party data)

This project evaluates models on **BLEnD**:

> Junho Myung, Nayeon Lee, Yi Zhou, et al. (2024). *BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse
> Cultures and Languages.* Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track),
> pp. 78104–78146. doi:10.52202/079017-2483
> Data: https://huggingface.co/datasets/nayeon212/BLEnD, code: https://github.com/nlee0212/BLEnD

BLEnD is released under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
license (https://creativecommons.org/licenses/by-sa/4.0/). The BLEnD files themselves are **not**
redistributed here: `run_all.sh` downloads them from the official repository (commit `7b9c131`).

## Files in this repository that contain BLEnD-derived content

The following files include BLEnD questions, prompts or reference answers, together with our model responses
and scores. As adaptations of BLEnD they are released under **CC BY-SA 4.0** as well:

| Path | BLEnD content |
|---|---|
| `results/*/raw/*.jsonl` | prompts (BLEnD questions + `inst-4` template) and model responses |
| `results/*/scored/*.csv` | matched BLEnD reference answers, model responses, item scores |
| `results/*/analysis/response_language_items.csv` | model responses per question id |
| `results/manual/*/sample.csv`, `results/manual/*/reviewers/*` | BLEnD questions and reference answers with human judgments |
| `results/manual/*/error_analysis_candidates.csv` | BLEnD questions and reference answers |

Changes made to the BLEnD material: selection of the Azerbaijan and Iran short-answer questions,
questions sent to the models with the `inst-4` prompt, and model responses scored against the annotations.

## Other data in this repository

- **Model responses**: generated with GPT-4.1 (OpenAI) and Gemini 3.5 Flash-Lite (Google) in September 2026.
  They are shared for reproducibility.
- **Human judgments** (`results/manual/*/reviewers/`): anonymised. Reviewers are identified only as A and B,
  and the workbooks contain no names in their metadata.
- **Aggregate tables and figures** (`results/comparison/`, `results/manual/validation_summary.*`,
  `results/figures/`): CC BY-SA 4.0.

## Third-party code

`third_party/az_stemmer/` is the Azerbaijani stemmer from aznlp-disc/stemmer (MIT license, see its `LICENSE.md`),
the same stemmer BLEnD's evaluation uses.
