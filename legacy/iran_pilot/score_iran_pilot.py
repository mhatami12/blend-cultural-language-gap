import json
import os
import re
import string
import sys
import unicodedata as ud
from pathlib import Path

import pandas as pd
import spacy
from hazm import Lemmatizer


def clean_response(response, prompt):
    response = (response or "").replace(prompt, "")
    response = response.replace("：", ":").replace("、", ",").replace("，", ",").replace("。", ".")
    response = response.strip().lower()
    match = re.findall(r"^(\w+:)\s", response)
    extracted = ""
    for m in match:
        if len(m) > len(extracted) and m.replace(":", "") in prompt.lower():
            extracted = m
    if match:
        response = response.replace(extracted, "").strip()
    return response.strip(".")


def normalize_tokens(tokens):
    d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
    return [
        ud.normalize("NFD", t).translate(d).lower()
        for t in tokens
        if t not in string.punctuation and t != ""
    ]


def lemma_check(answer, response, language, en_nlp, fa_lemma):
    if not answer or not response:
        return False

    if (
        answer in response
        or answer.replace("-", " ") in response
        or answer.replace(" ", "-") in response
    ):
        return True

    if language == "English":
        answer_tokens = [t.lemma_ for t in en_nlp(answer)]
        response_tokens = [t.lemma_ for t in en_nlp(response)]
    elif language == "Persian":
        answer_tokens = [fa_lemma.lemmatize(term) for term in answer.split()]
        response_tokens = [fa_lemma.lemmatize(term) for term in response.split()]
    else:
        raise ValueError(language)

    answer_tokens = normalize_tokens(answer_tokens)
    response_tokens = normalize_tokens(response_tokens)

    return all(a in response_tokens for a in answer_tokens)


def score_file(path, annotations):
    with path.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]

    en_nlp = spacy.load("en_core_web_sm")
    fa_lemma = Lemmatizer()

    rows = []
    for r in recs:
        ann = annotations[r["id"]]
        valid = not (
            ann["idks"]["no-answer"] + ann["idks"]["not-applicable"] >= 3
            or ann["idks"]["idk"] >= 5
            or len(ann["annotations"]) == 0
        )

        response = clean_response(r["response"], r["prompt"])
        binary = 0
        weight = 0.0
        matched = None

        if valid and response and ann["annotations"]:
            max_vote = ann["annotations"][0]["count"]

            # Exact BLEnD logic: local-language references first for
            # non-English responses, then English references as fallback.
            for agg in ann["annotations"]:
                if r["language"] != "English":
                    for a in agg["answers"]:
                        if lemma_check(a, response, r["language"], en_nlp, fa_lemma):
                            binary = 1
                            weight = agg["count"] / max_vote
                            matched = a
                            break
                if binary:
                    break

                for a in agg["en_answers"]:
                    if lemma_check(a, response, "English", en_nlp, fa_lemma):
                        binary = 1
                        weight = agg["count"] / max_vote
                        matched = a
                        break

                if binary:
                    break

        rows.append({
            "id": r["id"],
            "topic": r["topic"],
            "language": r["language"],
            "model_key": r["model_key"],
            "valid": valid,
            "binary_score": binary,
            "weight_score": weight,
            "matched_answer": matched,
            "response_clean": response,
            "response_raw": r["response"],
        })

    return pd.DataFrame(rows)


def main():
    root = Path(".")
    blend = root / "BLEnD"
    ann_path = blend / "data" / "annotations" / "Iran_data.json"
    with ann_path.open(encoding="utf-8") as f:
        annotations = json.load(f)

    raw = root / "results" / "iran_pilot" / "raw"
    scored = root / "results" / "iran_pilot" / "scored"
    scored.mkdir(parents=True, exist_ok=True)

    summary = []

    for path in sorted(raw.glob("*.jsonl")):
        df = score_file(path, annotations)
        out = scored / f"{path.stem}.csv"
        df.to_csv(out, index=False, encoding="utf-8")

        v = df[df["valid"]].copy()
        sem_b = 100 * v["binary_score"].mean() if len(v) else float("nan")
        sem_w = 100 * v["weight_score"].mean() if len(v) else float("nan")

        summary.append({
            "model": df["model_key"].iloc[0],
            "language": df["language"].iloc[0],
            "n_saved": len(df),
            "n_valid": len(v),
            "SEM-B": sem_b,
            "SEM-W": sem_w,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        root / "results" / "iran_pilot" / "summary.csv",
        index=False,
        encoding="utf-8"
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
