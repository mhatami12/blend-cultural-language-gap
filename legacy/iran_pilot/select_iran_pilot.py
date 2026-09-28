import argparse
import json
from pathlib import Path

import pandas as pd


DOMAINS = [
    "Food",
    "Holidays/Celebration/Leisure",
    "Sport",
    "Education",
    "Work life",
    "Family",
]

def is_valid(ann):
    idks = ann.get("idks", {})
    return not (
        idks.get("no-answer", 0) + idks.get("not-applicable", 0) >= 3
        or idks.get("idk", 0) >= 5
        or len(ann.get("annotations", [])) == 0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-dir", default=None,
                    help="Path to the cloned BLEnD repository.")
    ap.add_argument("--output", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.blend_dir) if args.blend_dir else Path("BLEnD")
    q_path = base / "data" / "questions" / "Iran_questions.csv"
    a_path = base / "data" / "annotations" / "Iran_data.json"

    if not q_path.exists():
        raise FileNotFoundError(f"Missing: {q_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Missing: {a_path}")

    q = pd.read_csv(q_path)
    with open(a_path, encoding="utf-8") as f:
        ann = json.load(f)

    q["valid"] = q["ID"].map(lambda x: is_valid(ann[x]))
    qv = q[q["valid"]].copy()

    # Even allocation across the six domains.
    base_n = args.n // len(DOMAINS)
    remainder = args.n % len(DOMAINS)
    targets = {
        d: base_n + (1 if i < remainder else 0)
        for i, d in enumerate(DOMAINS)
    }

    rng = args.seed
    selected = []
    counts = {}

    for domain in DOMAINS:
        pool = qv[qv["Topic"] == domain].sample(frac=1, random_state=rng)
        take = min(targets[domain], len(pool))
        chosen = pool.head(take)
        selected.append(chosen)
        counts[domain] = take
        rng += 1

    out = pd.concat(selected, ignore_index=True)

    # If a domain had too few valid items, fill the remaining slots from
    # remaining valid questions while preserving determinism.
    if len(out) < args.n:
        remaining = qv[~qv["ID"].isin(out["ID"])]
        remaining = remaining.sample(frac=1, random_state=999)
        out = pd.concat([out, remaining.head(args.n - len(out))], ignore_index=True)

    out = out[["ID", "Topic", "Source", "Question", "Translation"]]
    out.to_csv(args.output or "selected_iran_pilot_questions.csv",
               index=False, encoding="utf-8-sig")

    print("Selected:", len(out))
    print("\nBy topic:")
    print(out["Topic"].value_counts().reindex(DOMAINS).fillna(0).astype(int))
    print("\nOutput:", args.output or "selected_iran_pilot_questions.csv")


if __name__ == "__main__":
    main()
