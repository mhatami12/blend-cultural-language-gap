"""Exploratory: in which language / script do the models answer LOCAL-language prompts?

A model that answers an Azerbaijani or Persian question in English (or Turkish, Russian)
is still scored against the English references by BLEnD, so a language switch can hide
or create part of the gap. This script measures how often that happens.

Short answers (often 1-3 words, many proper nouns such as 'Novruz' or 'plov') are hard to
classify, so low-confidence cases are labelled 'uncertain' instead of being forced.

  Azerbaijan: Azerbaijani | English | Turkish | Russian/Cyrillic | uncertain
              (Cyrillic -> Russian; the letter 'ə' -> Azerbaijani; otherwise the lingua
              detector with confidence >= 0.75 if installed, else 'uncertain')
  Iran:       Persian script | Latin script | mixed | uncertain
              (Perso-Arabic letters vs Latin letters)

Outputs: results/comparison/response_language.csv (share and accuracy per detected language)
         results/<Country>/analysis/response_language_items.csv (per response, for spot checks)

Design
  ResponseLanguageClassifier  one strategy per local language (open for new languages)
  ResponseLanguageAnalysis    applies the classifier of each setting to its scored responses

Usage:
    python -m blend.response_language
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import pandas as pd

from .config import COUNTRIES, Paths, model_name


class ResponseLanguageClassifier(ABC):
    """Labels the language/script of one response to a local-language prompt."""
    language: str  # the local language whose prompts this classifier handles

    def classify(self, text) -> str:
        text = "" if text is None else str(text).strip()
        if not text or text == "nan":
            return "empty"
        return self._classify(text)

    @abstractmethod
    def _classify(self, text: str) -> str:
        ...


class AzerbaijaniResponseClassifier(ResponseLanguageClassifier):
    """Cyrillic -> Russian; the letter 'ə' -> Azerbaijani; otherwise the lingua detector with
    confidence >= MIN_CONF (if installed), else 'uncertain'. One-word answers are 'uncertain'."""
    language = "Azerbaijani"
    CYRILLIC = re.compile(r"[\u0400-\u04FF]")
    MIN_CONF = 0.75

    def __init__(self, detector=None, use_lingua: bool = True):
        self._names = {}
        self.detector = detector
        if detector is None and use_lingua:
            try:  # optional dependency (pip install lingua-language-detector)
                from lingua import Language, LanguageDetectorBuilder
                self._names = {Language.AZERBAIJANI: "Azerbaijani", Language.ENGLISH: "English",
                               Language.TURKISH: "Turkish", Language.RUSSIAN: "Russian/Cyrillic"}
                self.detector = LanguageDetectorBuilder.from_languages(*self._names).build()
            except ImportError:  # pragma: no cover
                self.detector = None

    def _classify(self, text: str) -> str:
        if self.CYRILLIC.search(text):
            return "Russian/Cyrillic"
        if "ə" in text.lower():  # schwa is Azerbaijani Latin script, not Turkish
            return "Azerbaijani"
        if len(re.findall(r"\w+", text)) < 2 or self.detector is None:  # names/dishes: not classifiable
            return "uncertain"
        top = self.detector.compute_language_confidence_values(text)[0]
        return self._names.get(top.language, str(top.language)) if top.value >= self.MIN_CONF else "uncertain"


class PersianResponseClassifier(ResponseLanguageClassifier):
    """Perso-Arabic letters vs Latin letters."""
    language = "Persian"
    ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
    LATIN = re.compile(r"[A-Za-z\u00C0-\u024F]")

    def _classify(self, text: str) -> str:
        fa, lat = bool(self.ARABIC_SCRIPT.search(text)), bool(self.LATIN.search(text))
        if fa and lat:
            return "mixed"
        if fa:
            return "Persian script"
        if lat:
            return "Latin script"
        return "uncertain"  # digits only, punctuation


def default_classifiers() -> dict[str, ResponseLanguageClassifier]:
    return {c.language: c for c in (AzerbaijaniResponseClassifier(), PersianResponseClassifier())}


class ResponseLanguageAnalysis:
    def __init__(self, paths: Paths, classifiers: dict[str, ResponseLanguageClassifier]):
        self.paths = paths
        self.classifiers = classifiers

    def classify_country(self, country: str) -> pd.DataFrame | None:
        lang = COUNTRIES[country].language
        frames = []
        for path in sorted(self.paths.scored(country).glob(f"*_{lang}.csv")):
            df = pd.read_csv(path)
            df = df[df.valid].copy()
            df["response_language"] = df.response_raw.map(self.classifiers[lang].classify)
            frames.append(df)
        return pd.concat(frames) if frames else None

    def run(self) -> pd.DataFrame:
        self.paths.comparison.mkdir(parents=True, exist_ok=True)
        rows = []
        for country in COUNTRIES:
            items = self.classify_country(country)
            if items is None:
                continue
            out = self.paths.analysis(country)
            out.mkdir(parents=True, exist_ok=True)
            items[["model_key", "id", "topic", "response_raw", "response_language", "binary_score"]].to_csv(
                out / "response_language_items.csv", index=False, encoding="utf-8")
            for (m, rl), g in items.groupby(["model_key", "response_language"]):
                n_model = (items.model_key == m).sum()
                rows.append({"country": country, "model": model_name(m), "response_language": rl, "n": len(g),
                             "share_pct": 100 * len(g) / n_model, "accuracy_pct": 100 * g.binary_score.mean()})
        res = pd.DataFrame(rows)
        res.to_csv(self.paths.comparison / "response_language.csv", index=False, float_format="%.2f")
        return res


def main():
    classifiers = default_classifiers()
    res = ResponseLanguageAnalysis(Paths.from_env(), classifiers).run()
    print("Language/script of answers to LOCAL-language prompts:\n")
    print(res.round(1).to_string(index=False))
    if classifiers["Azerbaijani"].detector is None:
        print("\n(lingua not installed: multi-word Azerbaijani-prompt answers without 'ə' are 'uncertain')")
    print("\nSpot-check ~30 rows of results/<Country>/analysis/response_language_items.csv before reporting.")


if __name__ == "__main__":
    main()
