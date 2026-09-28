"""Study configuration as immutable value objects (single source of truth).

Everything that differs between the two BLEnD settings is a `CountrySpec`; everything that
must be identical across them (prompt, models, generation settings, statistics) is defined
once here. Classes receive what they need from this module through their constructors
(dependency injection), so tests can pass their own `Paths` or settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# BLEnD commit the experiment was run against (run_all.sh checks it out; test_all.sh verifies it)
BLEND_COMMIT = "7b9c131719e7fe5f9bed0f8b855532d613cc9f2b"
PROMPT_ID = "inst-4"  # official BLEnD direct-answer instruction; first row wins, as in BLEnD's make_prompt()
ENGLISH = "English"

DOMAIN_ORDER = ("Food", "Holidays/Celebration/Leisure", "Sport", "Education", "Work life", "Family")
DOMAIN_SHORT = {"Holidays/Celebration/Leisure": "Holidays/Leisure"}


@dataclass(frozen=True)
class CountrySpec:
    """One BLEnD culture-language setting."""
    name: str          # as in BLEnD file names, e.g. "Iran"
    language: str      # local language as in BLEnD's evaluation code, e.g. "Persian"
    code: str          # ISO code, e.g. "fa"

    @property
    def languages(self) -> tuple[str, str]:
        return ENGLISH, self.language


@dataclass(frozen=True)
class ModelSpec:
    key: str           # file-name key, e.g. "gpt-4.1"
    provider: str      # "openai" | "google"
    api_model: str
    display_name: str


@dataclass(frozen=True)
class GenerationSettings:
    """Defaults reproduce the reported runs (poster v7):
       GPT-4.1: temperature 0, seed 42, max 128 tokens
       Gemini:  provider-default temperature/thinking (no argument sent), max 128 tokens
    Do not re-run inference for the reported results. GEMINI_TEMPERATURE=0 gives a separate,
    deterministic Gemini run, which must be reported as a different setup."""
    temperature: float = 0.0
    max_tokens: int = 128
    seed: int = 42
    gemini_temperature: str = "default"   # "default" (as reported) or a number

    @classmethod
    def from_env(cls) -> "GenerationSettings":
        return cls(gemini_temperature=os.environ.get("GEMINI_TEMPERATURE", "default"))


@dataclass(frozen=True)
class StatsSettings:
    seed: int = 42
    n_boot: int = 10_000
    n_perm: int = 10_000
    alpha: float = 0.05


@dataclass(frozen=True)
class Paths:
    """Where input data and results live. Read from the environment at call time, so
    BLEND_DIR / RESULTS_DIR set by a caller (tests, run_all.sh) are respected.
    The default dataset folder is external/BLEnD: a top-level 'BLEnD' folder would collide with the
    code package 'blend' on case-insensitive file systems (macOS, Windows)."""
    blend_dir: Path
    results: Path
    third_party: Path = field(default=ROOT / "third_party")

    @classmethod
    def from_env(cls) -> "Paths":
        return cls(blend_dir=Path(os.environ.get("BLEND_DIR", ROOT / "external" / "BLEnD")),
                   results=Path(os.environ.get("RESULTS_DIR", ROOT / "results")))

    def raw(self, country: str) -> Path:
        return self.results / country / "raw"

    def scored(self, country: str) -> Path:
        return self.results / country / "scored"

    def analysis(self, country: str) -> Path:
        return self.results / country / "analysis"

    @property
    def comparison(self) -> Path:
        return self.results / "comparison"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def manual(self) -> Path:
        return self.results / "manual"


COUNTRIES: dict[str, CountrySpec] = {
    "Azerbaijan": CountrySpec("Azerbaijan", "Azerbaijani", "az"),
    "Iran": CountrySpec("Iran", "Persian", "fa"),
}

MODELS: dict[str, ModelSpec] = {
    "gpt-4.1": ModelSpec("gpt-4.1", "openai", "gpt-4.1", "GPT-4.1"),
    "gemini-3.5-flash-lite": ModelSpec("gemini-3.5-flash-lite", "google", "gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
}
MODEL_ORDER: tuple[str, ...] = ("gpt-4.1", "gemini-3.5-flash-lite")

STATS = StatsSettings()


def model_name(key: str) -> str:
    """Display name for a model key (unknown keys are returned unchanged)."""
    return MODELS[key].display_name if key in MODELS else key
