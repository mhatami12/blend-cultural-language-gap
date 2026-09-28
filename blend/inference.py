"""Query the models with BLEnD short-answer questions (resumable).

Every response is appended to results/<Country>/raw/<model>_<language>.jsonl as soon as it
arrives, with the exact prompt and the generation settings used. Re-running the same command
skips IDs that are already saved, so an interrupted run simply continues. Questions that still
fail after retries are skipped (not saved) and picked up on the next run.

Design
  ModelClient      abstraction of an LLM API (OpenAIClient, GeminiClient; tests use a fake)
  RetryPolicy      exponential back-off, separate from the clients
  ResponseStore    the JSONL file of one model x language (also used by the scorer)
  InferenceRunner  the loop; receives dataset, client and store (dependency injection)

Usage:
    python -m blend.inference --country Iran --model gpt-4.1
    python -m blend.inference --country all --model all
    python -m blend.inference --country Azerbaijan --model all --limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from .config import COUNTRIES, ENGLISH, MODEL_ORDER, MODELS, GenerationSettings, ModelSpec, Paths
from .data import BlendDataset


@dataclass(frozen=True)
class Generation:
    """One model answer plus the metadata stored next to it."""
    text: str
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------ clients
class ModelClient(ABC):
    def __init__(self, spec: ModelSpec, settings: GenerationSettings):
        self.spec = spec
        self.settings = settings

    @abstractmethod
    def generate(self, prompt: str) -> Generation:
        ...


class OpenAIClient(ModelClient):
    def __init__(self, spec: ModelSpec, settings: GenerationSettings):
        super().__init__(spec, settings)
        from openai import OpenAI
        self._client = OpenAI()  # reads OPENAI_API_KEY

    def generate(self, prompt: str) -> Generation:
        r = self._client.chat.completions.create(
            model=self.spec.api_model, messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.temperature, max_tokens=self.settings.max_tokens, seed=self.settings.seed)
        c = r.choices[0]
        return Generation(c.message.content or "", {
            "finish_reason": c.finish_reason, "model_version": r.model,
            "system_fingerprint": r.system_fingerprint, "temperature": self.settings.temperature})


class GeminiClient(ModelClient):
    def __init__(self, spec: ModelSpec, settings: GenerationSettings):
        super().__init__(spec, settings)
        from google import genai
        from google.genai import types
        self._client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        kwargs = {"max_output_tokens": settings.max_tokens}
        if settings.gemini_temperature != "default":
            kwargs["temperature"] = float(settings.gemini_temperature)
        self._temperature = kwargs.get("temperature", "provider-default")
        self._config = types.GenerateContentConfig(**kwargs)

    def generate(self, prompt: str) -> Generation:
        r = self._client.models.generate_content(model=self.spec.api_model, contents=prompt, config=self._config)
        finish = r.candidates[0].finish_reason.name if r.candidates else "NO_CANDIDATE"
        return Generation(r.text or "", {"finish_reason": finish,
                                         "model_version": getattr(r, "model_version", self.spec.api_model),
                                         "temperature": self._temperature})


CLIENTS: dict[str, type[ModelClient]] = {"openai": OpenAIClient, "google": GeminiClient}


def make_client(spec: ModelSpec, settings: GenerationSettings) -> ModelClient:
    """Factory: pick the client class for the model's provider."""
    return CLIENTS[spec.provider](spec, settings)


# ------------------------------------------------------------------ retries
@dataclass
class RetryPolicy:
    tries: int = 6
    max_wait: int = 60
    sleep: Callable[[float], None] = time.sleep

    def call(self, fn: Callable[[], Generation]) -> Optional[Generation]:
        for attempt in range(self.tries):
            try:
                return fn()
            except Exception as e:  # rate limits, timeouts, 5xx
                wait = min(self.max_wait, 2 ** attempt)
                print(f"  error ({type(e).__name__}: {str(e)[:120]}), retry in {wait}s")
                self.sleep(wait)
        return None


# ------------------------------------------------------------------ storage
class ResponseStore:
    """The JSONL file with the responses of one model in one language."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def for_run(cls, paths: Paths, country: str, model_key: str, language: str) -> "ResponseStore":
        return cls(paths.raw(country) / f"{model_key}_{language}.jsonl")

    def records(self) -> Iterator[dict]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def latest_by_id(self) -> dict[str, dict]:
        """BLEnD keeps the LAST response per question id."""
        return {r["id"]: r for r in self.records()}

    def done_ids(self) -> set[str]:
        return {r["id"] for r in self.records()}

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ runner
class InferenceRunner:
    def __init__(self, dataset: BlendDataset, client: ModelClient, store: ResponseStore,
                 retry: RetryPolicy | None = None):
        self.dataset = dataset
        self.client = client
        self.store = store
        self.retry = retry or RetryPolicy()

    def run(self, language: str, all_questions: bool = False, limit: int | None = None) -> int:
        """Query every not-yet-saved question; returns the number of new responses."""
        questions = self.dataset.questions if all_questions else self.dataset.valid_questions
        todo = questions[~questions["ID"].isin(self.store.done_ids())]
        if limit:
            todo = todo.head(limit)
        country, model = self.dataset.country.name, self.client.spec.key
        print(f"{country} / {model} / {language}: {len(todo)} to go -> {self.store.path}")
        saved = failed = 0
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            prompt = self.dataset.build_prompt(row, language)
            gen = self.retry.call(lambda: self.client.generate(prompt))
            if gen is None:
                failed += 1
                continue
            self.store.append({"id": row["ID"], "topic": str(row["Topic"]), "country": country, "language": language,
                               "model_key": model, "api_model": self.client.spec.api_model, "prompt": prompt,
                               "response": gen.text, **gen.meta, "timestamp": datetime.now(timezone.utc).isoformat()})
            saved += 1
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
        if failed:
            print(f"  {failed} questions failed after retries; re-run the same command to retry them.")
        return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="all", help=f"{list(COUNTRIES)} or all")
    ap.add_argument("--model", default="all", help=f"{list(MODELS)} or all")
    ap.add_argument("--language", default="all", help="English, local, or all")
    ap.add_argument("--all-questions", action="store_true",
                    help="also query questions that BLEnD excludes (default: valid questions only)")
    ap.add_argument("--limit", type=int, help="only the first N remaining questions (smoke test)")
    a = ap.parse_args()

    paths, settings = Paths.from_env(), GenerationSettings.from_env()
    countries = list(COUNTRIES) if a.country == "all" else [a.country]
    models = list(MODEL_ORDER) if a.model == "all" else [a.model]
    for c in countries:
        if c not in COUNTRIES:
            raise SystemExit(f"unknown country {c}")
        spec = COUNTRIES[c]
        langs = {"all": list(spec.languages), ENGLISH: [ENGLISH], "local": [spec.language],
                 spec.language: [spec.language]}[a.language]
        dataset = BlendDataset(spec, paths.blend_dir)
        for m in models:
            if m not in MODELS:
                raise SystemExit(f"unknown model {m}")
            client = make_client(MODELS[m], settings)
            for lang in langs:
                InferenceRunner(dataset, client, ResponseStore.for_run(paths, c, m, lang)).run(lang, a.all_questions, a.limit)


if __name__ == "__main__":
    main()
