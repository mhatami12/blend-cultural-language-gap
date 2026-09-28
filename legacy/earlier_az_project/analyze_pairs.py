"""Analyze paired correctness after running the official BLEnD evaluator.

Expected input files in work/responses/:
  {model}-Azerbaijan_Azerbaijani_inst-4_response_score.csv
  {model}-Azerbaijan_English_inst-4_response_score.csv

These files are produced by BLEnD/evaluation/evaluate.py, which adds binary_score per item.
"""
import os
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESP = WORK / "responses"
OUT = WORK / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

models = [m for m in [os.environ.get("MODEL_A"), os.environ.get("MODEL_B")] if m]
if not models:
    # Fallback: infer model names from the files.
    models = sorted({p.name.split("-Azerbaijan_")[0] for p in RESP.glob("*-Azerbaijan_English_inst-4_response_score.csv")})
if not models:
    raise SystemExit("No scored response files found.")

summary = []
all_pairs = []

for model in models:
    az_path = RESP / f"{model}-Azerbaijan_Azerbaijani_inst-4_response_score.csv"
    en_path = RESP / f"{model}-Azerbaijan_English_inst-4_response_score.csv"
    if not az_path.exists() or not en_path.exists():
        print(f"Skipping {model}: missing one or both scored files.")
        continue

    az = pd.read_csv(az_path, encoding="utf-8")[["ID", "binary_score", "response"]].rename(columns={"binary_score":"az_correct", "response":"az_response"})
    en = pd.read_csv(en_path, encoding="utf-8")[["ID", "binary_score", "response"]].rename(columns={"binary_score":"en_correct", "response":"en_response"})
    pair = az.merge(en, on="ID", how="inner")
    pair = pair.dropna(subset=["az_correct", "en_correct"]).copy()
    pair["az_correct"] = pair["az_correct"].astype(int)
    pair["en_correct"] = pair["en_correct"].astype(int)
    pair["model"] = model
    pair["pattern"] = "both_wrong"
    pair.loc[(pair.en_correct==1)&(pair.az_correct==1), "pattern"] = "both_correct"
    pair.loc[(pair.en_correct==1)&(pair.az_correct==0), "pattern"] = "english_only"
    pair.loc[(pair.en_correct==0)&(pair.az_correct==1), "pattern"] = "azerbaijani_only"

    n = len(pair)
    en_acc = pair.en_correct.mean() * 100
    az_acc = pair.az_correct.mean() * 100
    b = int(((pair.en_correct==1)&(pair.az_correct==0)).sum())
    c = int(((pair.en_correct==0)&(pair.az_correct==1)).sum())

    # McNemar exact test: under H0, discordant cases are equally likely in either direction.
    if b+c > 0:
        p = binomtest(min(b,c), n=b+c, p=0.5, alternative='two-sided').pvalue
    else:
        p = 1.0

    summary.append({
        "model": model,
        "n_paired": n,
        "english_accuracy_pct": round(en_acc, 2),
        "azerbaijani_accuracy_pct": round(az_acc, 2),
        "gap_english_minus_azerbaijani_pp": round(en_acc-az_acc, 2),
        "both_correct_pct": round(((pair.pattern=="both_correct").mean()*100),2),
        "english_only_pct": round(((pair.pattern=="english_only").mean()*100),2),
        "azerbaijani_only_pct": round(((pair.pattern=="azerbaijani_only").mean()*100),2),
        "both_wrong_pct": round(((pair.pattern=="both_wrong").mean()*100),2),
        "mcnemar_exact_p": p,
        "discordant_pairs": b+c,
    })

    pair.to_csv(OUT / f"{model}_paired_results.csv", index=False, encoding="utf-8-sig")
    pair[pair.pattern.isin(["english_only","azerbaijani_only"])].to_csv(OUT / f"{model}_discordant_candidates.csv", index=False, encoding="utf-8-sig")
    all_pairs.append(pair)

if summary:
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))
else:
    print("No complete model results found.")
