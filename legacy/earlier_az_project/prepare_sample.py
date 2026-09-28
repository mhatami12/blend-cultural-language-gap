"""Prepare a reproducible 300-question Azerbaijani BLEnD sample.

Run from a clone of the official BLEnD repository, or provide BLEND_DIR.
Example:
  set BLEND_DIR=C:\\path\\to\\BLEnD
  python prepare_sample.py
"""
import json
import os
from pathlib import Path
import pandas as pd

SEED = 42
N = 300
blend_dir = Path(os.environ.get("BLEND_DIR", "../BLEnD")).resolve()
out_dir = Path(__file__).resolve().parent / "work"
out_dir.mkdir(parents=True, exist_ok=True)

q_path = blend_dir / "data" / "questions" / "Azerbaijan_questions.csv"
a_path = blend_dir / "data" / "annotations" / "Azerbaijan_data.json"
prompt_path = blend_dir / "data" / "prompts" / "Azerbaijan_prompts.csv"

for p in [q_path, a_path, prompt_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing BLEnD file: {p}")

q = pd.read_csv(q_path, encoding="utf-8")
with a_path.open("r", encoding="utf-8") as f:
    annotations = json.load(f)

if len(q) != 500:
    raise ValueError(f"Expected 500 Azerbaijan questions in original BLEnD; found {len(q)}")

# Stratified sample by topic; this preserves topic representation while remaining reproducible.
parts = []
for topic, g in q.groupby("topic", dropna=False):
    parts.append(g.sample(frac=N/len(q), random_state=SEED))
sample = pd.concat(parts, ignore_index=False)

# Exact total may be off by one due to rounding across topics; fix deterministically.
if len(sample) > N:
    sample = sample.sample(n=N, random_state=SEED)
elif len(sample) < N:
    remaining = q.loc[~q["ID"].isin(sample["ID"])]
    sample = pd.concat([sample, remaining.sample(n=N-len(sample), random_state=SEED)])

sample = sample.sort_values("ID").reset_index(drop=True)
ids = set(sample["ID"].astype(str))
subset_annotations = {k: v for k, v in annotations.items() if k in ids}

sample.to_csv(out_dir / "Azerbaijan_sample_300.csv", index=False, encoding="utf-8-sig")
with (out_dir / "Azerbaijan_300_data.json").open("w", encoding="utf-8") as f:
    json.dump(subset_annotations, f, ensure_ascii=False, indent=2)

prompts = pd.read_csv(prompt_path, encoding="utf-8")
prompts[prompts["id"].eq("inst-4")].to_csv(out_dir / "Azerbaijan_inst4_prompt.csv", index=False, encoding="utf-8-sig")

print(f"Prepared {len(sample)} paired questions.")
print("Topic counts:")
print(sample["topic"].value_counts().to_string())
print(f"Outputs written to: {out_dir}")
