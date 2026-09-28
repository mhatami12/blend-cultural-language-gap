"""Read-only access to the official BLEnD files (questions, prompts, annotations)."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd

from .config import DOMAIN_ORDER, ENGLISH, PROMPT_ID, CountrySpec

Annotation = dict  # one entry of BLEnD's <Country>_data.json


def is_valid_question(ann: Annotation) -> bool:
    """Official BLEnD exclusion rule (evaluation/exact_match.py, soft_exact_match)."""
    idks = ann.get("idks", {})
    return not (
        idks.get("no-answer", 0) + idks.get("not-applicable", 0) >= 3
        or idks.get("idk", 0) >= 5
        or len(ann.get("annotations", [])) == 0
    )


def reference_answers(ann: Annotation, key: str, k: int = 5) -> str:
    """Top-k annotated answers (first spelling of each), for review sheets."""
    return " | ".join(a for g in ann["annotations"][:k] for a in g[key][:1])


class BlendDataset:
    """The BLEnD data of one culture-language setting. Files are read lazily, once."""

    def __init__(self, country: CountrySpec, blend_dir: Path):
        self.country = country
        self.blend_dir = Path(blend_dir)

    def _file(self, *parts: str) -> Path:
        path = self.blend_dir.joinpath("data", *parts)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}\nClone BLEnD first: git clone https://github.com/nlee0212/BLEnD.git "
                                    "(or set BLEND_DIR)")
        return path

    @cached_property
    def questions(self) -> pd.DataFrame:
        """All paired questions. In BLEnD's CSVs 'Question' is the local language and
        'Translation' is the English version."""
        df = pd.read_csv(self._file("questions", f"{self.country.name}_questions.csv"))
        return df[["ID", "Topic", "Question", "Translation"]]

    @cached_property
    def annotations(self) -> dict[str, Annotation]:
        with open(self._file("annotations", f"{self.country.name}_data.json"), encoding="utf-8") as f:
            return json.load(f)

    @cached_property
    def prompt_templates(self) -> dict[str, str]:
        p = pd.read_csv(self._file("prompts", f"{self.country.name}_prompts.csv"))
        row = p[p["id"] == PROMPT_ID].iloc[0]  # Azerbaijan has inst-4 twice; BLEnD uses the first
        return {ENGLISH: row["English"], self.country.language: row["Translation"]}

    @cached_property
    def valid_questions(self) -> pd.DataFrame:
        q = self.questions
        q = q[q["ID"].map(lambda i: is_valid_question(self.annotations[i]))].copy()
        q["Topic"] = pd.Categorical(q["Topic"], list(DOMAIN_ORDER), ordered=True)
        return q.sort_values(["Topic", "ID"]).reset_index(drop=True)

    def annotation(self, question_id: str) -> Annotation:
        return self.annotations[question_id]

    def build_prompt(self, row, language: str) -> str:
        """Same as BLEnD's make_prompt(): only the language of prompt + question changes."""
        question = row["Translation"] if language == ENGLISH else row["Question"]
        return self.prompt_templates[language].replace("{q}", question)
