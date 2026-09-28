"""Create hand-annotation sheets for one or both countries (one-time step).

1. results/manual/<Country>/sample_new.csv
   N local-language responses (default 50), stratified by the AUTOMATIC label
   (half scored correct, half incorrect, split as evenly as possible across models), so
   that both missed correct answers and falsely accepted answers can be estimated.
   Reviewers fill `human_judgment` with Correct / Incorrect / Uncertain.
   To use it with `blend.validation`, rename it to sample.csv and put each reviewer's
   copy in results/manual/<Country>/reviewers/.
   The existing sample.csv (the one that was reviewed) is never overwritten.

2. results/manual/<Country>/error_analysis_candidates.csv
   All language-dependent cases (correct in one language, wrong in the other) with the
   reference answers, sorted by domain, for qualitative error analysis. Suggested
   `error_category` labels: wrong_cultural_answer | generic | wrong_entity_or_context |
   incomplete | language_problem | list_answer | uncertain

Design
  ManualSampler  builds both sheets for one setting from its scored responses; dataset and
                 paths are injected

Usage:
    python -m blend.manual_sample                     # both countries
    python -m blend.manual_sample --country Iran --n 50
    python -m blend.manual_sample --errors-only       # (run_all.sh) only the error-analysis sheets
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import COUNTRIES, MODEL_ORDER, STATS, Paths, model_name
from .data import BlendDataset, reference_answers


class ManualSampler:
    def __init__(self, dataset: BlendDataset, paths: Paths, seed: int = STATS.seed):
        self.dataset = dataset
        self.paths = paths
        self.seed = seed
        self.country = dataset.country.name
        self.lang = dataset.country.language

    def _ref(self, question_id: str, key: str) -> str:
        return reference_answers(self.dataset.annotation(question_id), key)

    def _scored(self, model: str, language: str) -> pd.DataFrame | None:
        path = self.paths.scored(self.country) / f"{model}_{language}.csv"
        return pd.read_csv(path) if path.exists() else None

    def validation_sample(self, n_total: int = 50) -> pd.DataFrame:
        """n_total local-language responses, half auto-correct / half auto-incorrect, split across models."""
        dfs = {m: d[d.valid] for m in MODEL_ORDER if (d := self._scored(m, self.lang)) is not None}
        if not dfs:
            raise SystemExit(f"{self.country}: run `python -m blend.score` first")
        parts = []
        for label in (1, 0):
            need = n_total // 2 if label == 1 else n_total - n_total // 2
            quotas = {m: need // len(dfs) + (1 if i < need % len(dfs) else 0) for i, m in enumerate(dfs)}
            taken = {}
            for i, (m, d) in enumerate(dfs.items()):
                pool = d[d.binary_score == label]
                taken[m] = pool.sample(min(quotas[m], len(pool)), random_state=self.seed + i + 100 * label)
            short = need - sum(len(t) for t in taken.values())
            for i, (m, d) in enumerate(dfs.items()):  # top up from the other model(s) if one ran short
                if short <= 0:
                    break
                pool = d[(d.binary_score == label) & ~d.id.isin(taken[m].id)]
                extra = pool.sample(min(short, len(pool)), random_state=self.seed + 1000 + i + 100 * label)
                taken[m] = pd.concat([taken[m], extra])
                short -= len(extra)
            parts += list(taken.values())
        s = pd.concat(parts).sample(frac=1, random_state=self.seed)  # shuffle so labels are not in blocks
        ann = self.dataset.annotation
        lang = self.lang.lower()
        return pd.DataFrame({
            "id": s.id, "topic": s.topic, "model": s.model_key.map(model_name),
            f"question_{lang}": s.id.map(lambda i: ann(i)["question"]),
            "question_english": s.id.map(lambda i: ann(i)["en_question"]),
            "llm_response": s.response_raw,
            f"reference_{lang}": s.id.map(lambda i: self._ref(i, "answers")),
            "reference_english": s.id.map(lambda i: self._ref(i, "en_answers")),
            "automatic_score": s.binary_score, "human_judgment": "", "reviewer_note": "",
        })

    def error_candidates(self) -> pd.DataFrame:
        """All questions answered correctly in exactly one of the two languages."""
        rows = []
        for m in MODEL_ORDER:
            en, lo = self._scored(m, "English"), self._scored(m, self.lang)
            if en is None or lo is None:
                continue
            d = en[en.valid].merge(lo[lo.valid], on=["id", "topic", "model_key"], suffixes=("_en", "_loc"))
            d = d[d.binary_score_en != d.binary_score_loc].copy()
            d["direction"] = d.binary_score_en.map({1: "English only correct", 0: f"{self.lang} only correct"})
            rows.append(d)
        err = pd.concat(rows)
        ann = self.dataset.annotation
        lang = self.lang.lower()
        return pd.DataFrame({
            "model": err.model_key.map(model_name), "id": err.id, "topic": err.topic, "direction": err.direction,
            "question_english": err.id.map(lambda i: ann(i)["en_question"]),
            "response_english": err.response_raw_en, f"response_{lang}": err.response_raw_loc,
            "reference_english": err.id.map(lambda i: self._ref(i, "en_answers")),
            f"reference_{lang}": err.id.map(lambda i: self._ref(i, "answers")),
            "list_answer_en": err.is_list_en, f"list_answer_{lang}": err.is_list_loc,
            "error_category": "",
        }).sort_values(["topic", "direction", "model"])

    def write(self, n_total: int = 50, errors_only: bool = False) -> None:
        out = self.paths.manual / self.country
        out.mkdir(parents=True, exist_ok=True)
        e = self.error_candidates()
        e.to_csv(out / "error_analysis_candidates.csv", index=False, encoding="utf-8-sig")
        print(f"{self.country}: {len(e)} language-dependent cases -> {out / 'error_analysis_candidates.csv'}")
        if errors_only:
            return
        s = self.validation_sample(n_total)
        s.to_csv(out / "sample_new.csv", index=False, encoding="utf-8-sig")  # utf-8-sig opens cleanly in Excel
        print(f"{self.country}: {len(s)} rows -> {out / 'sample_new.csv'} "
              f"({(s.automatic_score == 1).sum()} auto-correct / {(s.automatic_score == 0).sum()} auto-incorrect)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="all")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--errors-only", action="store_true", help="only write error_analysis_candidates.csv")
    a = ap.parse_args()
    paths = Paths.from_env()
    for c in (COUNTRIES if a.country == "all" else [a.country]):
        ManualSampler(BlendDataset(COUNTRIES[c], paths.blend_dir), paths).write(a.n, a.errors_only)


if __name__ == "__main__":
    main()
