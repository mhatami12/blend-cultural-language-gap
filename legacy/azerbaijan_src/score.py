"""BLEnD short-answer scoring, trimmed to English and Azerbaijani.

This is a reduced copy of the official BLEnD scorer
(github.com/nlee0212/BLEnD, evaluation/exact_match.py -> soft_exact_match,
lemma_check; evaluation/evaluation_utils.py -> get_llm_response_by_id,
delete_prompt_from_answer). Only the English and Azerbaijani branches are kept,
so the heavy dependencies for the other 14 languages are not needed.
The matching logic itself is unchanged.

Usage:
    python src/score.py            # scores every file in results/raw/
"""
import json
import os
import re
import sys
import unicodedata as ud
from contextlib import contextmanager
from string import punctuation

import pandas as pd
import spacy

from common import ROOT, RAW_DIR, SCORED_DIR, load_annotations, is_valid_question

sys.path.insert(0, str(ROOT / "third_party"))
from az_stemmer.stemmer import Stemmer as AZStemmer  # noqa: E402  (aznlp-disc/stemmer, MIT)


@contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


with _cwd(ROOT / "third_party" / "az_stemmer"):  # stemmer loads words.txt / suffix.txt from cwd
    AZ_STEMMER = AZStemmer()
EN_NLP = spacy.load("en_core_web_sm")


# ------------------------------------------------ copied from evaluation_utils.py
def delete_prompt_from_answer(text, prompt):
    text = text.replace(prompt, '').replace('：', ':').replace('、', ',').replace('，', ',').replace('。', '.').lower()
    prompt = prompt.replace('：', ':').replace('、', ',').replace('，', ',').replace('。', '.').lower()
    match = re.findall(r'^(\w+:)\s', text)
    extracted = ''
    for m in match:
        if len(m) > len(extracted) and m.replace(':', '') in prompt:
            extracted = m
    if match:
        return text.replace(extracted, '').strip()
    return text.strip()


def clean_response(response, prompt):
    """get_llm_response_by_id(): remove prompt / 'Answer:' prefix, strip '.', lowercase."""
    return delete_prompt_from_answer(response, prompt).strip('.').lower()


# ------------------------------------------------ copied from exact_match.py
def _az_stem_words(my_text):
    my_text = my_text.replace("İ", "I").replace("“", "").replace("”", "").replace("'", "").replace('"', "")
    my_words = [''.join(c for c in w if (c not in punctuation) or (c == '-')) for w in my_text.split()]
    return AZ_STEMMER.stem_words(my_words)


def lemma_check(answer, llm_response, language):
    if answer in llm_response or answer.replace('-', ' ') in llm_response or answer.replace(' ', '-') in llm_response:
        return True
    if language == 'English':
        answer_tokens = [t.lemma_ for t in EN_NLP(answer)]
        llm_tokens = [t.lemma_ for t in EN_NLP(llm_response)]
    elif language == 'Azerbaijani':
        answer_tokens = _az_stem_words(answer)
        llm_tokens = _az_stem_words(llm_response)
    else:
        raise ValueError(language)
    d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
    answer_tokens = [ud.normalize('NFD', t).translate(d).lower() for t in answer_tokens if t not in punctuation and t != '']
    llm_tokens = [ud.normalize('NFD', t).translate(d).lower() for t in llm_tokens if t not in punctuation and t != '']
    return all(a in llm_tokens for a in answer_tokens)


def score_one(ann, llm_response, language):
    """Returns (binary, weight, matched_answer). Mirrors the soft_exact_match loop:
    for non-English queries, local-language answers are checked first, then English ones."""
    if not llm_response or not ann["annotations"]:
        return 0, 0.0, None
    max_vote = ann["annotations"][0]["count"]
    for agg in ann["annotations"]:
        if language != 'English':
            for a in agg["answers"]:
                if lemma_check(a, llm_response, language):
                    return 1, agg["count"] / max_vote, a
        for a in agg["en_answers"]:
            if lemma_check(a, llm_response, 'English'):
                return 1, agg["count"] / max_vote, a
    return 0, 0.0, None


# ------------------------------------------------ driver
def score_file(path, annotations):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = []
    for r in recs:
        ann = annotations[r["id"]]
        resp = clean_response(r["response"], r["prompt"])
        b, w, matched = score_one(ann, resp, r["language"])
        rows.append({
            "id": r["id"], "topic": r["topic"], "language": r["language"], "model_key": r["model_key"],
            "valid": is_valid_question(ann), "binary_score": b, "weight_score": w,
            "matched_answer": matched, "response_clean": resp, "response_raw": r["response"],
            "finish_reason": r.get("finish_reason"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    annotations = load_annotations()
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        sys.exit("no raw responses in results/raw/")
    for path in files:
        df = score_file(path, annotations)
        out = SCORED_DIR / (path.stem + ".csv")
        df.to_csv(out, index=False, encoding="utf-8")
        v = df[df.valid]
        print(f"{path.stem:40s} n_saved={len(df):3d}  n_valid={len(v):3d}  SEM-B={100*v.binary_score.mean():5.1f}  "
              f"SEM-W={100*v.weight_score.mean():5.1f}")
