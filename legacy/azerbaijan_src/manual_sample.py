"""Create the two hand-annotation sheets.

1. results/manual/manual_validation.csv
   50 Azerbaijani-prompt responses, stratified by the AUTOMATIC label
   (25 scored correct, 25 scored incorrect, split across models) so that both
   missed correct answers and falsely accepted answers can be estimated.
   Fill the column `human_correct` with 1 or 0 (leave empty if truly unsure and
   write why in `note`). analyze.py picks the file up automatically.

2. results/manual/error_analysis_candidates.csv
   All language-dependent cases (correct in one language, wrong in the other)
   with the reference answers, sorted by domain. Pick 20-30 from the domains with
   the largest gaps and fill `error_category` with one of:
   wrong_cultural_answer | generic | wrong_entity_or_context | incomplete |
   language_problem | uncertain

Usage:
    python src/manual_sample.py
"""
import pandas as pd

from common import RESULTS, SCORED_DIR, load_annotations

SEED = 42
N_TOTAL = 50


def ref_answers(ann, key, k=5):
    return " | ".join(a for g in ann["annotations"][:k] for a in g[key][:1])


if __name__ == "__main__":
    out_dir = RESULTS / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = load_annotations()

    az_files = [p for p in sorted(SCORED_DIR.glob("*_Azerbaijani.csv")) if not p.stem.startswith(("TEST", "SIM"))]
    if not az_files:
        raise SystemExit("run score.py first")

    # ---- 1. manual validation sheet
    # Allocate 25 rows to each automatic label across models, as evenly as possible.
    # With two models this yields 13/12 per label rather than 48 rows from integer division.
    n_models = len(az_files)
    target_per_label = N_TOTAL // 2
    base = target_per_label // n_models
    remainder = target_per_label % n_models
    parts = []
    for label in (1, 0):
        for i, p in enumerate(az_files):
            df = pd.read_csv(p)
            df = df[df.valid]
            pool = df[df.binary_score == label]
            target = base + (1 if i < remainder else 0)
            take = min(target, len(pool))
            if take:
                parts.append(pool.sample(take, random_state=SEED + i + label * 100))
        # If one model lacks enough items in a stratum, fill the remaining quota from the other models.
        current = sum(len(x) for x in parts[-n_models:] if len(x) > 0)
        remaining = target_per_label - current
        if remaining > 0:
            for i, p in enumerate(az_files):
                df = pd.read_csv(p)
                df = df[df.valid]
                already = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                used_ids = set(already.loc[already.get('binary_score', pd.Series(dtype=int)) == label, 'id']) if not already.empty and 'id' in already.columns and 'binary_score' in already.columns else set()
                pool = df[(df.binary_score == label) & (~df.id.isin(used_ids))]
                if pool.empty:
                    continue
                take = min(remaining, len(pool))
                parts.append(pool.sample(take, random_state=SEED + 1000 + i + label * 100))
                remaining -= take
                if remaining <= 0:
                    break
    val = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)  # shuffle so labels are not in blocks
    val["question_az"] = val.id.map(lambda i: ann[i]["question"])
    val["question_en"] = val.id.map(lambda i: ann[i]["en_question"])
    val["reference_az"] = val.id.map(lambda i: ref_answers(ann[i], "answers"))
    val["reference_en"] = val.id.map(lambda i: ref_answers(ann[i], "en_answers"))
    val = val.rename(columns={"binary_score": "auto_correct"})
    val["human_correct"] = ""
    val["note"] = ""
    cols = ["model_key", "id", "topic", "question_az", "question_en", "response_raw",
            "reference_az", "reference_en", "auto_correct", "human_correct", "note"]
    path = out_dir / "manual_validation.csv"
    if path.exists():
        print(f"{path} exists, not overwritten (delete it to resample)")
    else:
        val[cols].to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig opens cleanly in Excel
        print(f"wrote {len(val)} rows -> {path}")

    # ---- 2. error analysis candidates
    rows = []
    for p in az_files:
        model = p.stem.rsplit("_", 1)[0]
        az = pd.read_csv(p)
        en = pd.read_csv(SCORED_DIR / f"{model}_English.csv")
        m = en[en.valid].merge(az[az.valid], on=["id", "topic", "model_key"], suffixes=("_en", "_az"))
        d = m[m.binary_score_en != m.binary_score_az].copy()
        d["direction"] = d.binary_score_en.map({1: "English only correct", 0: "Azerbaijani only correct"})
        rows.append(d)
    err = pd.concat(rows)
    err["question_en"] = err.id.map(lambda i: ann[i]["en_question"])
    err["reference_en"] = err.id.map(lambda i: ref_answers(ann[i], "en_answers"))
    err["reference_az"] = err.id.map(lambda i: ref_answers(ann[i], "answers"))
    err["error_category"] = ""
    err = err.sort_values(["topic", "direction", "model_key"])
    cols = ["model_key", "id", "topic", "direction", "question_en", "response_raw_en", "response_raw_az",
            "reference_en", "reference_az", "error_category"]
    path = out_dir / "error_analysis_candidates.csv"
    err[cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(err)} language-dependent cases -> {path}")
