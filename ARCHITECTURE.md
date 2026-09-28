# Architecture

The package `blend/` is organised as small classes with one job each. Data flows in one direction:

```text
BLEnD files ─► BlendDataset ─► InferenceRunner ─► ResponseStore (raw/*.jsonl)
                                                        │
                    BlendScorer ◄── LanguageNormalizer  ▼
                         └──────► ResponseScorer / ScoringPipeline ─► scored/*.csv
                                                        │
                        ScoredResults ─► PairedScores ─► PairedAnalysis / BetweenSettingComparison
                                                        │
                   AnalysisTables ─► SummaryWriter (summary.md), save_tables (*.csv), PosterFigures (fig*.pdf)
reviewer sheets ─► JudgmentSheet ─► CountryValidation ─► ValidationReport (validation_summary.md)
```

## Classes per module

| Module | Classes | Responsibility |
|---|---|---|
| `config.py` | `CountrySpec`, `ModelSpec`, `GenerationSettings`, `StatsSettings`, `Paths` | immutable settings (frozen dataclasses) |
| `data.py` | `BlendDataset` | read-only access to the BLEnD files of one setting (lazy, cached) |
| `inference.py` | `ModelClient` (abstract), `OpenAIClient`, `GeminiClient`, `RetryPolicy`, `ResponseStore`, `InferenceRunner` | querying models, resumable storage |
| `scoring.py` | `LanguageNormalizer` (abstract), `EnglishLemmatizer`, `AzerbaijaniStemmer`, `PersianLemmatizer`, `ResponseCleaner`, `AnswerExtractor`, `BlendScorer`, `ScoreResult` | BLEnD matching logic |
| `score.py` | `ResponseScorer`, `ScoringPipeline` | raw responses → scored tables |
| `stats.py` | `PairedScores`, `PairedAnalysis`, `BetweenSettingComparison` | RQ1–RQ3 statistics |
| `analyze.py` | `ScoredResults`, `StudyAnalysis`, `AnalysisTables`, `SummaryWriter` | tables and summary |
| `figures.py` | `Palette`, `PosterFigures` | poster figures |
| `validation.py` | `SheetReader`, `JudgmentSheet`, `CountryValidation`, `ValidationReport` | manual validation |
| `response_language.py` | `ResponseLanguageClassifier` (abstract), `AzerbaijaniResponseClassifier`, `PersianResponseClassifier`, `ResponseLanguageAnalysis` | language of the answers |
| `manual_sample.py` | `ManualSampler` | review sheets |
| `formatting.py` | `format_p` | shared presentation helper |

Every module keeps a small `main()` so that `python -m blend.<module>` works as before.

## SOLID in this code

| Principle | Where |
|---|---|
| **S**ingle responsibility | Data access (`ScoredResults`), computation (`StudyAnalysis`), presentation (`SummaryWriter`) and persistence (`save_tables`) are separate. `RetryPolicy` is separate from the API clients, and `ResponseStore` from the runner. |
| **O**pen/closed | A new language needs only a new `LanguageNormalizer` subclass, without changing `BlendScorer` (see `test_new_language_plugs_in_without_changing_the_scorer`). The same holds for a new `ModelClient` (register it in `CLIENTS`) and a new `ResponseLanguageClassifier`. |
| **L**iskov substitution | Any `ModelClient` works in `InferenceRunner`; the tests replace the real API with `FakeClient`. Any `LanguageNormalizer` works in `BlendScorer`. |
| **I**nterface segregation | The abstract interfaces are one method each: `ModelClient.generate()`, `LanguageNormalizer.tokens()`, `ResponseLanguageClassifier.classify()`. |
| **D**ependency inversion | High-level classes receive their dependencies through the constructor and depend on abstractions: `InferenceRunner(dataset, client, store, retry)`, `BlendScorer(normalizers)`, `CountryValidation(country, folder)`, `PosterFigures(overall, domains, out_dir)`. Tests inject temporary folders and fakes, so nothing touches the real results or an API. |

## Other clean-code rules followed

- **Immutable configuration.** Settings are frozen dataclasses. `Paths.from_env()` reads `BLEND_DIR` and `RESULTS_DIR` at call time, not at import.
- **Value objects** with validation: `PairedScores` checks its columns; `ScoreResult` replaces anonymous tuples.
- **No hidden global state.**
  - Every bootstrap and permutation uses its own seeded generator, so results do not depend on call order (tested).
  - Language tools are loaded lazily per instance (`cached_property`).
- **Named constants** instead of magic values, for example:
  - `AnswerExtractor.MAX_INTRO_WORDS`
  - `AzerbaijaniResponseClassifier.MIN_CONF`
  - `StatsSettings.alpha`
- **Type hints** on all public methods, and docstrings that state what each method returns and why.
- **Behaviour-preserving refactor.** All 25 result tables, all figures and all 3,700 item scores are byte-identical before and after the OOP refactor. `./test_all.sh` checks this.
