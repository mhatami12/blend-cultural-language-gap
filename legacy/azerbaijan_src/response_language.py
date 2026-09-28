"""Exploratory: in which language/script does the model answer Azerbaijani prompts?

Short answers (often 1-3 words, many proper nouns such as 'Novruz' or 'plov') are
hard to classify, so low-confidence cases are labelled 'uncertain' instead of forced.
Categories: Azerbaijani, English, Turkish, Russian/Cyrillic, uncertain.
One-word answers without a script cue (Cyrillic, or the Azerbaijani letter ə) are 'uncertain'.

Usage:
    python src/response_language.py
"""
import re

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

from common import ANALYSIS_DIR, SCORED_DIR

LANGS = [Language.AZERBAIJANI, Language.ENGLISH, Language.TURKISH, Language.RUSSIAN]
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGS).build()
NAMES = {Language.AZERBAIJANI: "Azerbaijani", Language.ENGLISH: "English",
         Language.TURKISH: "Turkish", Language.RUSSIAN: "Russian/Cyrillic"}
CYRILLIC = re.compile(r"[\u0400-\u04FF]")
MIN_CONF = 0.75


def classify(text: str) -> str:
    text = str(text).strip()
    if not text or text == "nan":
        return "empty"
    if CYRILLIC.search(text):
        return "Russian/Cyrillic"
    if "ə" in text.lower():  # schwa is used in Azerbaijani Latin script, not in Turkish
        return "Azerbaijani"
    if len(re.findall(r"\w+", text)) < 2:  # one-word answers (often names/dishes) are not classifiable
        return "uncertain"
    conf = DETECTOR.compute_language_confidence_values(text)
    top = conf[0]
    return NAMES[top.language] if top.value >= MIN_CONF else "uncertain"


if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(SCORED_DIR.glob("*_Azerbaijani.csv")):
        df = pd.read_csv(path)
        df = df[df.valid]
        df["response_language"] = df["response_raw"].map(classify)
        frames.append(df)
    if not frames:
        raise SystemExit("run score.py first")
    all_df = pd.concat(frames)
    all_df[["model_key", "id", "topic", "response_raw", "response_language", "binary_score"]].to_csv(
        ANALYSIS_DIR / "response_language_items.csv", index=False, encoding="utf-8")

    share = (all_df.groupby("model_key")["response_language"].value_counts(normalize=True)
             .mul(100).round(1).unstack(fill_value=0))
    acc = (all_df.groupby(["model_key", "response_language"])["binary_score"].mean()
           .mul(100).round(1).unstack())
    share.to_csv(ANALYSIS_DIR / "response_language_share.csv")
    acc.to_csv(ANALYSIS_DIR / "response_language_accuracy.csv")
    print("Share of Azerbaijani-prompt responses by detected language (%):\n", share, "\n")
    print("Accuracy by detected response language (%):\n", acc)
    print("\nNote: spot-check ~30 rows of response_language_items.csv by hand before reporting.")
