"""Score every raw response file of one or both countries.

Writes results/<Country>/scored/<model>_<language>.csv with, per question:
  binary_score / weight_score   official BLEnD SEM-B / SEM-W item scores
  strict_score                  sensitivity score: first answer only (see scoring.AnswerExtractor)
  n_words, is_list              response length and whether it contained several answers

Usage:
    python -m blend.score                    # both countries
    python -m blend.score --country Iran
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import COUNTRIES, Paths
from .data import BlendDataset, is_valid_question
from .inference import ResponseStore
from .scoring import AnswerExtractor, BlendScorer, ResponseCleaner


class ResponseScorer:
    """Turns stored responses into a per-question score table (official + strict)."""

    def __init__(self, dataset: BlendDataset, scorer: BlendScorer):
        self.dataset = dataset
        self.scorer = scorer

    def score_record(self, r: dict) -> dict:
        ann = self.dataset.annotation(r["id"])
        lang = r["language"]
        resp = ResponseCleaner.clean(r["response"], r["prompt"])
        first = AnswerExtractor.first_answer(resp)
        official = self.scorer.score(ann, resp, lang)
        strict = self.scorer.score(ann, first, lang) if first != resp else official
        return {
            "id": r["id"], "topic": r["topic"], "country": self.dataset.country.name, "language": lang,
            "model_key": r["model_key"], "valid": is_valid_question(ann),
            "binary_score": official.binary, "weight_score": official.weight, "matched_answer": official.matched,
            "strict_score": strict.binary, "first_answer": first,
            "n_words": len((r["response"] or "").split()), "is_list": AnswerExtractor.is_multi_answer(resp),
            "response_clean": resp, "response_raw": r["response"],
            "finish_reason": r.get("finish_reason"), "model_version": r.get("model_version"),
        }

    def score_file(self, path: Path) -> pd.DataFrame:
        return pd.DataFrame([self.score_record(r) for r in ResponseStore(path).latest_by_id().values()])


class ScoringPipeline:
    """Scores all raw files of one country and writes the scored CSVs."""

    def __init__(self, country: str, paths: Paths, scorer: BlendScorer):
        self.country = country
        self.paths = paths
        self.scorer = ResponseScorer(BlendDataset(COUNTRIES[country], paths.blend_dir), scorer)

    def run(self) -> dict[str, pd.DataFrame]:
        raw_dir, out_dir = self.paths.raw(self.country), self.paths.scored(self.country)
        files = sorted(raw_dir.glob("*.jsonl"))
        if not files:
            print(f"{self.country}: no raw responses in {raw_dir}")
            return {}
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for path in files:
            df = self.scorer.score_file(path)
            df.to_csv(out_dir / f"{path.stem}.csv", index=False, encoding="utf-8")
            v = df[df.valid]
            print(f"{self.country:10s} {path.stem:36s} saved={len(df):3d} valid={len(v):3d}  "
                  f"SEM-B={100 * v.binary_score.mean():5.1f}  SEM-W={100 * v.weight_score.mean():5.1f}  "
                  f"strict={100 * v.strict_score.mean():5.1f}  lists={100 * v.is_list.mean():4.1f}%")
            results[path.stem] = df
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="all")
    a = ap.parse_args()
    paths = Paths.from_env()
    scorer = BlendScorer.default(paths.third_party)
    for c in (COUNTRIES if a.country == "all" else [a.country]):
        ScoringPipeline(c, paths, scorer).run()


if __name__ == "__main__":
    main()
