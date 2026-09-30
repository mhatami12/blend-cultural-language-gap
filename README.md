# Does Cultural Domain Affect the Cross-Lingual Performance Gap of LLMs?
### Azerbaijani and Persian on the BLEnD benchmark: unified pipeline

One codebase for both culture-language settings (Azerbaijan/Azerbaijani, Iran/Persian).
It replaces `src/`, `iran_pilot/` and the Colab-only analysis/figure code. Everything on
the poster can be regenerated from the saved model responses with one command:

```bash
./setup.sh            # requirements + pinned hazm + spaCy model
./run_all.sh          # clones BLEnD (pinned commit) if needed, then score -> analyze -> validation -> figures
pytest -q             # 57 tests: unit tests per class + reproduction of every poster-v7 number
./test_all.sh         # everything: environment, rebuild, pytest, old-vs-new scorer, inference (43 checks)
```

See **TESTING.md** to test every class step by step in IPython, **ARCHITECTURE.md** for the class design (OOP/SOLID),
and **VERIFY.md** for a step-by-step check that the changes were applied correctly.

## Layout

```text
blend/                 (classes and SOLID: see ARCHITECTURE.md)
  config.py      frozen settings: CountrySpec, ModelSpec, GenerationSettings, StatsSettings, Paths
  data.py        BlendDataset: official BLEnD questions / prompts / annotations, validity rule
  inference.py   ModelClient (OpenAI, Gemini), RetryPolicy, ResponseStore, InferenceRunner
                 python -m blend.inference --country Iran --model gpt-4.1
  scoring.py     BlendScorer + LanguageNormalizer strategies (EN/AZ/FA), ResponseCleaner, AnswerExtractor
  score.py       ResponseScorer, ScoringPipeline: raw -> scored CSVs      python -m blend.score
  stats.py       PairedScores, PairedAnalysis (RQ1/RQ2), BetweenSettingComparison (RQ3)
  analyze.py     ScoredResults, StudyAnalysis, SummaryWriter              python -m blend.analyze
  validation.py  JudgmentSheet, CountryValidation, ValidationReport       python -m blend.validation
  figures.py     Palette, PosterFigures                                   python -m blend.figures
  response_language.py  ResponseLanguageClassifier strategies             python -m blend.response_language
  manual_sample.py      ManualSampler                                     python -m blend.manual_sample
  formatting.py  shared p-value formatting
results/
  <Country>/raw/*.jsonl      model responses (the only expensive artefact; keep them)
  <Country>/scored/*.csv     per-question scores
  comparison/                overall.csv, domains.csv, interaction.csv, rq3_country_difference.csv,
                             response_style.csv, response_language.csv, summary.md  <- poster numbers
  manual/<Country>/          sample.csv + reviewers/*  -> validation_summary.md;
                             error_analysis_candidates.csv (language-dependent cases)
  figures/                   fig1_overall, fig2_domains, fig3_robustness, fig4_paired
third_party/az_stemmer/      Azerbaijani stemmer used by BLEnD (MIT)
tests/                       pytest: unit tests per class, validation workflow, poster-v7 reproduction
tools/test_all.py            one-command test of everything (run via ./test_all.sh)
legacy/                      old scripts, unchanged (for comparison only)
  azerbaijan_src/            the Azerbaijan scripts as they were run
  iran_pilot/                the Iran scripts as they were run
  earlier_az_project/        superseded 300-question version
  poster_pilot/              superseded 23-question pilot
```

## Design

| Item | Setting |
|---|---|
| Data | BLEnD `data/` (not `data_SemEval/`), commit `7b9c131`; Azerbaijan 469 / Iran 456 valid questions |
| Prompt | official `inst-4`, same template in English and the local language |
| Models | GPT-4.1 (`gpt-4.1-2025-04-14`), Gemini 3.5 Flash-Lite |
| Responses | 3,824 stored (Azerbaijan 2,000: all 500 questions; Iran 1,824: 456 valid questions), 3,700 evaluable |
| Generation | same BLEnD prompt and evaluation procedure across languages; generation settings recorded per response (GPT-4.1: temperature 0; Gemini: provider default) |
| Primary metric | SEM-B: official BLEnD soft-exact-match scoring, reproduced with an explicit safeguard for empty reference answers; SEM-W secondary |
| Gap | **English − local** (pp) everywhere; negative = local language better |
| RQ1 | exact McNemar, Wilson CIs, paired bootstrap CI of the gap |
| RQ2 | GEE logistic `correct ~ language × domain` clustered by question (primary); per-domain McNemar with Holm; discordant-pair homogeneity (permutation) |
| RQ3 | between-setting comparison of the gap (descriptive: settings differ in language, culture and questions), bootstrap CI + permutation p |
| Manual validation | 50 local-language responses per setting, stratified by automatic label (25 correct / 25 incorrect) and model, **not** by domain |
| Sensitivity analysis | `strict` score: only the first answer of a list counts (official stays primary) |

## Changes compared with the earlier scripts (and why)

1. **One pipeline for both countries.** Inference, scoring and statistics were duplicated
   (`src/` vs `iran_pilot/`) and had drifted apart; the Iran statistics and all poster
   figures only existed in Colab. Now every number comes from the same code.
2. **Reproduces the v7 poster exactly** (all accuracies, McNemar p, GEE χ² for both
   countries) when run on the saved responses with `hazm==0.10.0`.
3. **`hazm` version checked.** hazm 0.9.x lemmatises some Persian words differently (e.g. کارمندی),
   which flips 2 Iran/Gemini items and moves GEE χ² from 25.3 to 21.5. hazm 0.10.0 (installed by
   `setup.sh`) reproduces the reported scores exactly; an independent re-run found the same with 0.12.1.
   `test_all.sh` checks the lemmatiser's behaviour, not the version number.
4. **Empty reference answers are ignored.** BLEnD's Iran file has one empty English
   reference (`New-gr-08`); the official scorer would count *any* response as correct.
   The old Iran scorer already guarded this; the rule is now shared and documented.
5. **Strict (first-answer) score, reported as a SENSITIVITY analysis.** `official` (the BLEnD
   scorer) stays the primary metric. `inst-4` asks for a single answer, but models often return lists,
   and the official scorer accepts a list if *any* item matches. Listing rates differ by language
   (e.g. Iran/Gemini: 16% of Persian vs 2% of English responses; Azerbaijan/GPT-4.1: 16% of English vs 8%
   of Azerbaijani). `strict` credits only the first answer; a leading phrase such as "In Iran, …" is not
   taken as the answer. `results/comparison/summary.md` starts with a table of which conclusions hold
   under both scorings. Robust: the Azerbaijani advantage for Gemini; domain dependence for Gemini in
   both countries; no domain dependence for Iran/GPT-4.1. Three of four overall language effects depend
   on whether list answers are credited.
6. **Generation settings recorded per response; defaults = reported setup.** The reported runs
   used temperature 0 for GPT-4.1 and Gemini's provider-default temperature and thinking (no
   temperature/thinking argument was sent, in both countries). `config.py` reproduces exactly that
   by default. **Do not re-run inference for the reported results.** `GEMINI_TEMPERATURE=0` gives
   a deterministic Gemini run, which is a *different* setup and must be reported as such.
   Every saved record stores the temperature actually used.
7. **Inference queries only valid questions** by default (saves ~8% of API calls; excluded
   questions never enter the analysis anyway). `--all-questions` restores the old behaviour.
8. **Manual validation is computed, not typed in.** Both settings now have two independent reviewers
   (`results/manual/<Country>/reviewers/`). The earlier single-annotator pass for Azerbaijan (v2) remains
   only as the `human_correct` column of `results/manual/Azerbaijan/sample.csv` and is not used in any statistic.
   Reviewer sheets (`Correct / Incorrect / Uncertain`) are read directly (the old `analyze.py` expected 0/1 and
   silently skipped them); items are matched on id + response text. Results:
   - Iran: κ = .392 (3 categories) / .611 (both confident); automatic "correct" confirmed 17/17;
     automatic "incorrect" judged correct 8/11 → the scorer is conservative.
   - Azerbaijan: κ = .228 / .290; automatic "correct" confirmed 19/19; automatic "incorrect" judged
     correct 3/6. The single v2 annotator alone had suggested 11/13.
   - Azerbaijan reviewer A was replaced on 2026-09-30 by a new independent reviewer. The replaced sheet is
     kept in `results/manual/Azerbaijan/superseded/` for transparency and is not used in any statistic.
   Empty reviewer sheets are reported as *pending* and ignored.
9. **Bootstrap/permutation use their own seeded RNG per call**, so results no longer depend
   on the order in which models are analysed.
10. **New figures** (dumbbell with CIs, per-domain dot plot with CIs and Holm significance,
    robustness plot, paired outcomes), colour-blind-safe palette, one sign convention.
    The v7 line chart plotted Azerbaijan/GPT-4.1 Education at 0 instead of +7.2 pp.

## Adding or re-running data

```bash
export OPENAI_API_KEY=... GEMINI_API_KEY=...
python -m blend.inference --country Iran --model all --limit 3   # smoke test
python -m blend.inference --country all --model all              # resumable; re-run after interruptions
./run_all.sh
```

Raw responses from the provider APIs are not exactly reproducible (model updates, non-zero
temperature); keep `results/*/raw/` with the repository.

## License and attribution

- **BLEnD** (Myung et al., NeurIPS 2024; https://huggingface.co/datasets/nayeon212/BLEnD) is licensed under
  **CC BY-SA 4.0**. BLEnD itself is not included in this repository; `run_all.sh` downloads it.
- Files under `results/` that contain BLEnD questions or reference answers are adaptations of BLEnD and are
  therefore also released under **CC BY-SA 4.0**. `DATA_LICENSE.md` lists these files.
- `third_party/az_stemmer/` is under the MIT license (see its `LICENSE.md`).
- No API keys are stored in this repository. Keys are read from environment variables (see `.env.example`),
  and `.env` is git-ignored.

If you use this code or these results, please cite BLEnD (official citation from the BLEnD repository):

```bibtex
@inproceedings{NEURIPS2024_8eb88844,
 author = {Myung, Junho and Lee, Nayeon and Zhou, Yi and Jin, Jiho and Putri, Rifki Afina and Antypas, Dimosthenis and Borkakoty, Hsuvas and Kim, Eunsu and Perez-Almendros, Carla and Ayele, Abinew Ali and Guti\'{e}rrez-Basulto, V\'{\i}ctor and Ib\'{a}\~{n}ez-Garc\'{\i}a, Yazm\'{\i}n and Lee, Hwaran and Muhammad, Shamsuddeen Hassan and Park, Kiwoong and Rzayev, Anar Sabuhi and White, Nina and Yimam, Seid Muhie and Pilehvar, Mohammad Taher and Ousidhoum, Nedjma and Camacho-Collados, Jose and Oh, Alice},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/079017-2483},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {78104--78146},
 publisher = {Curran Associates, Inc.},
 title = {BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/8eb88844dafefa92a26aaec9f3acad93-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {37},
 year = {2024}
}
```

