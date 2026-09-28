# Testing the code step by step in Python / IPython

Each step tests one class. Copy a block into IPython and compare the output with the value after `# ->`.
Nothing here calls an API or changes files in `results/`.

## 0. Start

```bash
cd blend-crosslingual        # the repository root: `blend` must be importable from here
./setup.sh                   # once
ipython
```

```python
%load_ext autoreload
%autoreload 2                 # code edits apply without restarting IPython
import os; os.environ["BLEND_DIR"] = "../BLEnD"   # before creating Paths
import pandas as pd
```

## 1. Configuration: `config.py`

```python
from blend.config import COUNTRIES, MODELS, Paths, GenerationSettings
paths = Paths.from_env(); paths                      # -> blend_dir=../BLEnD, results=.../results
COUNTRIES["Iran"]                                    # -> CountrySpec(name='Iran', language='Persian', code='fa')
COUNTRIES["Iran"].languages                          # -> ('English', 'Persian')
GenerationSettings.from_env()                        # -> temperature=0.0, gemini_temperature='default' (= reported runs)
COUNTRIES["Iran"].language = "x"                     # -> FrozenInstanceError: settings are immutable
```

## 2. BLEnD data: `BlendDataset`

```python
from blend.data import BlendDataset
iran = BlendDataset(COUNTRIES["Iran"], paths.blend_dir)
az = BlendDataset(COUNTRIES["Azerbaijan"], paths.blend_dir)
len(iran.valid_questions), len(az.valid_questions)   # -> (456, 469)
row = iran.valid_questions.iloc[0]
print(iran.build_prompt(row, "Persian"))             # the official inst-4 prompt in Persian
print(iran.build_prompt(row, "English"))
iran.annotation("New-gr-08")["annotations"][1]       # -> en_answers contains '' (empty reference)
```

## 3. Scoring: `ResponseCleaner`, `AnswerExtractor`, `BlendScorer`

```python
from blend.scoring import BlendScorer, ResponseCleaner, AnswerExtractor
scorer = BlendScorer.default()
ResponseCleaner.clean("Answer: Plov.", "Question: x\nAnswer:")      # -> 'plov'
scorer.matches("sweet", "sweets", "English")                        # -> True (lemmatised)
AnswerExtractor.first_answer("فلافل، سمبوسه، بلال")                   # -> 'فلافل'
AnswerExtractor.first_answer("in iran, new year's eve is celebrated")   # -> the intro phrase is not the answer
scorer.score(iran.annotation("New-gr-08"), "anything", "English")   # -> ScoreResult(binary=0, ...) (safeguard)
```

**Open/closed principle.** To add a language, add a class; `BlendScorer` stays unchanged:

```python
from blend.scoring import LanguageNormalizer, EnglishLemmatizer
class Toy(LanguageNormalizer):
    language = "Toy"
    def tokens(self, text): return text.upper().split()
BlendScorer({"English": EnglishLemmatizer(), "Toy": Toy()}).matches("b a", "a x b", "Toy")   # -> True
```

## 4. Scoring real responses: `ResponseScorer`

This is the most important check. Score the saved responses again and compare with the files made earlier in Colab:

```python
from blend.score import ResponseScorer
rs = ResponseScorer(iran, scorer)
new = rs.score_file(paths.raw("Iran") / "gemini-3.5-flash-lite_Persian.jsonl")   # ~20 s
round(100 * new[new.valid].binary_score.mean(), 2)                                # -> 65.13 (poster v7: 65.1)
old = pd.read_csv("../results/iran_pilot/scored/gemini-3.5-flash-lite_Persian.csv")  # the Colab output
m = new.merge(old, on="id", suffixes=("_new", "_old"))
(m.binary_score_new != m.binary_score_old).sum()                                  # -> 0 (no item differs)
```

For Azerbaijan the old file is `../results/scored/gpt-4.1_Azerbaijani.csv`. Use the `az` dataset and the
`paths.raw("Azerbaijan")` file of the same name.

## 5. Statistics: `PairedAnalysis`, `BetweenSettingComparison`

```python
from blend.analyze import ScoredResults
from blend.stats import PairedAnalysis, BetweenSettingComparison, PairedScores
results = ScoredResults(paths)
merged = results.load("Iran", "gemini-3.5-flash-lite")
s = PairedAnalysis(ScoredResults.pairs(merged, "official")).summary()
round(s["acc_en"], 1), round(s["acc_loc"], 1), round(s["gap_pp"], 1), round(s["mcnemar_p"], 3)
# -> (57.7, 65.1, -7.5, 0.015)      negative gap = Persian better
a = PairedAnalysis(ScoredResults.pairs(merged, "official"))
a.gee_interaction()["wald_chi2"]                                     # -> 25.33
a.domain_table()[["domain", "n", "gap_pp", "mcnemar_p_holm"]]        # Food -> -28.1
PairedAnalysis(ScoredResults.pairs(merged, "strict")).summary()["gap_pp"]   # -> -2.0 (sensitivity)
```

Between-setting comparison (RQ3, official scoring):

```python
az_pairs = ScoredResults.pairs(results.load("Azerbaijan", "gemini-3.5-flash-lite"), "official")
ir_pairs = ScoredResults.pairs(merged, "official")
BetweenSettingComparison(az_pairs, ir_pairs).difference()            # -> diff -2.1 pp, p_permutation 0.606 (n.s.)
```

A test with your own small example, where you can compute the result by hand:

```python
toy = PairedScores(pd.DataFrame({"id": list("abcde"), "topic": ["Food"] * 5,
                                 "en": [1, 1, 0, 0, 1], "loc": [1, 0, 1, 1, 1]}))
PairedAnalysis(toy).summary()["gap_pp"]                              # -> ≈ -20.0 (60% - 80%)
```

## 6. Manual validation: `CountryValidation`

```python
from blend.validation import CountryValidation
print("\n".join(CountryValidation("Iran", paths.manual / "Iran").run().lines))
# -> kappa = 0.392; confident 90.3%, kappa = 0.611; auto-correct 17/17; missed 8/11
print("\n".join(CountryValidation("Azerbaijan", paths.manual / "Azerbaijan").run().lines))
# -> 2 reviewers; kappa = 0.135 / 0.462; auto-correct 14/14; missed 1/6
```

## 7. Inference without an API: `InferenceRunner` with a fake client

**Dependency inversion.** The runner works with any `ModelClient`, so a fake can stand in for the API:

```python
import tempfile
from pathlib import Path
from blend.inference import ModelClient, Generation, InferenceRunner, ResponseStore, RetryPolicy
class FakeClient(ModelClient):
    def generate(self, prompt): return Generation("Plov", {"temperature": 0})
store = ResponseStore(Path(tempfile.mkdtemp()) / "raw.jsonl")          # temporary file, results/ untouched
runner = InferenceRunner(iran, FakeClient(MODELS["gpt-4.1"], GenerationSettings()), store,
                         RetryPolicy(sleep=lambda s: None))
runner.run("Persian", limit=2)      # -> 2
runner.run("Persian", limit=2)      # -> 2 (resumes with the NEXT two questions)
[r["id"] for r in store.records()]  # -> 4 different ids
```

## 8. Figures

```python
from blend.figures import PosterFigures
from IPython.display import Image
out = Path(tempfile.mkdtemp())
PosterFigures(pd.read_csv(paths.comparison / "overall.csv"), pd.read_csv(paths.comparison / "domains.csv"), out).draw_all()
Image(str(out / "fig3_robustness.png"))
```

## 9. All automated tests

```python
!pytest -q                        # -> 57 passed
!./test_all.sh                    # -> 43 passed, 0 failed, 1 warning (live API test skipped)
```

Useful while exploring:
- `ScoredResults.load??` shows the source of a method.
- `help(PairedAnalysis)` shows the docstrings.
- `ARCHITECTURE.md` explains how the classes fit together and how SOLID is applied.
