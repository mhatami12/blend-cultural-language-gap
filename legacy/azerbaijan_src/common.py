"""Shared configuration and BLEnD data loading."""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BLEND_DIR = Path(os.environ.get("BLEND_DIR", ROOT / "BLEnD"))
RESULTS = ROOT / "results"
RAW_DIR = RESULTS / "raw"
SCORED_DIR = RESULTS / "scored"
ANALYSIS_DIR = RESULTS / "analysis"
FIG_DIR = RESULTS / "figures"

COUNTRY = "Azerbaijan"
LANGUAGES = ["English", "Azerbaijani"]
PROMPT_ID = "inst-4"  # official BLEnD instruction prompt

# key -> (provider, API model string). Record the exact strings you ran in the README.
MODELS = {
    "gpt-4.1": ("openai", "gpt-4.1"),
    "gemini-3.5-flash-lite": ("google", "gemini-3.5-flash-lite"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-6"),  # optional third model
}

CORE_MODELS = ["gpt-4.1", "gemini-3.5-flash-lite"]  # "--model all" runs these

GEN = {"temperature": 0.0, "max_tokens": 128}


def load_questions() -> pd.DataFrame:
    """500 paired questions. 'Question' = Azerbaijani, 'Translation' = English."""
    df = pd.read_csv(BLEND_DIR / "data" / "questions" / f"{COUNTRY}_questions.csv")
    return df[["ID", "Topic", "Question", "Translation"]]


def load_prompt_templates() -> dict:
    p = pd.read_csv(BLEND_DIR / "data" / "prompts" / f"{COUNTRY}_prompts.csv")
    row = p[p["id"] == PROMPT_ID].iloc[0]
    return {"English": row["English"], "Azerbaijani": row["Translation"]}


def load_annotations() -> dict:
    with open(BLEND_DIR / "data" / "annotations" / f"{COUNTRY}_data.json", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question_row, language: str, templates: dict) -> str:
    """Same logic as BLEnD's make_prompt(): only the language changes."""
    q = question_row["Translation"] if language == "English" else question_row["Question"]
    return templates[language].replace("{q}", q)


def is_valid_question(ann: dict) -> bool:
    """Official BLEnD exclusion rule (soft_exact_match in evaluation/exact_match.py)."""
    idks = ann["idks"]
    return not (
        idks["no-answer"] + idks["not-applicable"] >= 3
        or idks["idk"] >= 5
        or len(ann["annotations"]) == 0
    )
