"""BLEnD short-answer scoring for English, Azerbaijani and Persian.

`official` reproduces BLEnD's soft-exact-match scoring (soft_exact_match / lemma_check,
github.com/nlee0212/BLEnD, evaluation/exact_match.py + evaluation_utils.py) for the three
languages used here, with the same order of checks (for local-language prompts the
local-language references are tried first, then the English references) and one explicit
safeguard: empty reference answers never match.

`strict` is an added SENSITIVITY score, not part of BLEnD: the prompt asks for a *single*
answer, but models sometimes return a list ("falafel, samosa, corn, ..."). The official
scorer accepts a response if ANY listed item matches, which rewards listing. `strict` scores
only the first answer (see `AnswerExtractor`). Everything else is identical to `official`.

Design
  LanguageNormalizer   one strategy per language (Strategy pattern; add a language by adding
                       a subclass, without touching the scorer: open/closed principle)
  ResponseCleaner      BLEnD's response clean-up
  AnswerExtractor      first answer of a list answer (strict score)
  BlendScorer          the matching logic; depends on the LanguageNormalizer abstraction
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata as ud
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from string import punctuation
from typing import Mapping, Optional

from .config import ENGLISH, ROOT

_ACUTE = {ord("\N{COMBINING ACUTE ACCENT}"): None}


# ------------------------------------------------------------------ language strategies
class LanguageNormalizer(ABC):
    """Turns a text into comparable tokens (lemmas or stems) for one language."""
    language: str

    @abstractmethod
    def tokens(self, text: str) -> list[str]:
        ...


class EnglishLemmatizer(LanguageNormalizer):
    language = ENGLISH

    @cached_property
    def _nlp(self):  # loaded on first use
        import spacy
        return spacy.load("en_core_web_sm")

    def tokens(self, text: str) -> list[str]:
        return [t.lemma_ for t in self._nlp(text)]


class PersianLemmatizer(LanguageNormalizer):
    language = "Persian"

    @cached_property
    def _lemmatizer(self):
        from hazm import Lemmatizer
        return Lemmatizer()

    def tokens(self, text: str) -> list[str]:
        return [self._lemmatizer.lemmatize(t) for t in text.split()]


@contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class AzerbaijaniStemmer(LanguageNormalizer):
    """aznlp-disc/stemmer (MIT), the stemmer BLEnD uses; it reads words.txt/suffix.txt from cwd."""
    language = "Azerbaijani"

    def __init__(self, third_party: Path = ROOT / "third_party"):
        self.third_party = Path(third_party)

    @cached_property
    def _stemmer(self):
        if str(self.third_party) not in sys.path:
            sys.path.insert(0, str(self.third_party))
        from az_stemmer.stemmer import Stemmer
        with _cwd(self.third_party / "az_stemmer"):
            return Stemmer()

    def tokens(self, text: str) -> list[str]:
        text = text.replace("İ", "I").replace("“", "").replace("”", "").replace("'", "").replace('"', "")
        words = ["".join(c for c in w if (c not in punctuation) or (c == "-")) for w in text.split()]
        return self._stemmer.stem_words(words)


# ------------------------------------------------------------------ response handling
class ResponseCleaner:
    """BLEnD's get_llm_response_by_id(): drop the prompt / 'Answer:' prefix, strip '.', lowercase."""

    @staticmethod
    def delete_prompt_from_answer(text: str, prompt: str) -> str:
        text = text.replace(prompt, "").replace("：", ":").replace("、", ",").replace("，", ",").replace("。", ".").lower()
        prompt = prompt.replace("：", ":").replace("、", ",").replace("，", ",").replace("。", ".").lower()
        match = re.findall(r"^(\w+:)\s", text)
        extracted = ""
        for m in match:
            if len(m) > len(extracted) and m.replace(":", "") in prompt:
                extracted = m
        if match:
            return text.replace(extracted, "").strip()
        return text.strip()

    @classmethod
    def clean(cls, response: Optional[str], prompt: str) -> str:
        return cls.delete_prompt_from_answer(response or "", prompt).strip(".").lower()


class AnswerExtractor:
    """Splits a (cleaned) response into answer segments.

    Separators: comma, Persian comma, semicolon, non-numeric slash, newline, or a standalone
    "or" / "یا" / "və ya" / "yaxud". Parenthesised asides are removed. A leading segment that
    is only an introductory phrase ("In Iran,", "در ایران،", "Azərbaycanda,") is merged with
    the next segment instead of being taken as the first answer."""

    SPLIT = re.compile(r"[,،;؛\n]|(?<!\d)/(?!\d)|\s(?:or|یا|və ya|yaxud|ya da)\s")
    PARENS = re.compile(r"\([^)]*\)|（[^）]*）")
    INTRO = re.compile(r"^(?:(?:in|for|at)\s+.{0,25}|در\s+.{0,20}|.{0,15}(?:iran|azerbaijan|ایران|azərbaycan|iranda)\S*"
                       r"|for example|e\.g|such as|typically|usually|generally|معمولاً|مثلاً|adətən|məsələn)$")
    MAX_INTRO_WORDS = 4

    @classmethod
    def _tidy(cls, text: str) -> str:
        return re.sub(r"\s+", " ", cls.PARENS.sub(" ", text)).strip(" .:-")

    @classmethod
    def segments(cls, clean_text: str) -> list[str]:
        segs = [x.strip(" .:-") for x in cls.SPLIT.split(cls._tidy(clean_text))]
        segs = [x for x in segs if x]
        while len(segs) > 1 and len(segs[0].split()) <= cls.MAX_INTRO_WORDS and cls.INTRO.match(segs[0]):
            segs = [segs[0] + " " + segs[1]] + segs[2:]
        return segs

    @classmethod
    def first_answer(cls, clean_text: str) -> str:
        segs = cls.segments(clean_text)
        return cls._tidy(segs[0]) if segs else ""

    @classmethod
    def is_multi_answer(cls, clean_text: str) -> bool:
        return len(cls.segments(clean_text)) > 1


# ------------------------------------------------------------------ BLEnD matching
@dataclass(frozen=True)
class ScoreResult:
    binary: int                   # SEM-B item score (0/1)
    weight: float                 # SEM-W item score (vote share of the matched answer)
    matched: Optional[str] = None  # the reference answer that matched


NO_MATCH = ScoreResult(0, 0.0, None)


class BlendScorer:
    """BLEnD soft exact match. Depends only on the LanguageNormalizer abstraction
    (dependency inversion); the concrete normalizers are injected."""

    def __init__(self, normalizers: Mapping[str, LanguageNormalizer]):
        if ENGLISH not in normalizers:
            raise ValueError("an English normalizer is required (English references are always checked)")
        self.normalizers = dict(normalizers)

    @classmethod
    def default(cls, third_party: Path = ROOT / "third_party") -> "BlendScorer":
        return cls({n.language: n for n in (EnglishLemmatizer(), AzerbaijaniStemmer(third_party), PersianLemmatizer())})

    @staticmethod
    def _normalise(tokens: list[str]) -> list[str]:
        return [ud.normalize("NFD", t).translate(_ACUTE).lower() for t in tokens if t not in punctuation and t != ""]

    def matches(self, answer: str, response: str, language: str) -> bool:
        """BLEnD lemma_check(): substring match, else all answer tokens occur in the response."""
        if not answer.strip():  # safeguard: "" would match every response (BLEnD Iran New-gr-08)
            return False
        if answer in response or answer.replace("-", " ") in response or answer.replace(" ", "-") in response:
            return True
        try:
            normalizer = self.normalizers[language]
        except KeyError:
            raise ValueError(f"no normalizer for {language}; available: {list(self.normalizers)}") from None
        a_tok = self._normalise(normalizer.tokens(answer))
        r_tok = self._normalise(normalizer.tokens(response))
        return all(a in r_tok for a in a_tok)

    def score(self, ann: dict, response_clean: str, language: str) -> ScoreResult:
        """Mirrors soft_exact_match(): the first matching answer group (by votes) wins."""
        if not response_clean or not ann["annotations"]:
            return NO_MATCH
        max_vote = ann["annotations"][0]["count"]
        for group in ann["annotations"]:
            if language != ENGLISH:
                for a in group["answers"]:
                    if self.matches(a, response_clean, language):
                        return ScoreResult(1, group["count"] / max_vote, a)
            for a in group["en_answers"]:
                if self.matches(a, response_clean, ENGLISH):
                    return ScoreResult(1, group["count"] / max_vote, a)
        return NO_MATCH
