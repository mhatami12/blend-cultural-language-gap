"""Run the paired English/Azerbaijani experiment.

Prerequisites:
  pip install openai pandas
  Set OPENAI_API_KEY and MODEL_A; MODEL_B is optional.

The script uses the official BLEnD inst-4 prompt wording saved by prepare_sample.py.
It creates response files compatible with the official BLEnD evaluation pipeline.
"""
import os
from pathlib import Path
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
SAMPLE = WORK / "Azerbaijan_sample_300.csv"
PROMPT = WORK / "Azerbaijan_inst4_prompt.csv"
OUT = WORK / "responses"
OUT.mkdir(parents=True, exist_ok=True)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set.")

models = [m for m in [os.environ.get("MODEL_A"), os.environ.get("MODEL_B")] if m]
if not models:
    raise SystemExit("Set MODEL_A (and optionally MODEL_B).")

sample = pd.read_csv(SAMPLE, encoding="utf-8-sig")
prompts = pd.read_csv(PROMPT, encoding="utf-8-sig").iloc[0]

# The official file stores English text in column 'English' and Azerbaijani in 'Translation'.
en_template = prompts["English"]
az_template = prompts["Translation"]

client = OpenAI(api_key=api_key)

for model in models:
    for language, question_col, template in [
        ("English", "Question", en_template),
        ("Azerbaijani", "Translation", az_template),
    ]:
        rows = []
        out_file = OUT / f"{model}-Azerbaijan_{language}_inst-4_result.csv"
        existing = {}
        if out_file.exists():
            prev = pd.read_csv(out_file, encoding="utf-8")
            existing = dict(zip(prev["ID"].astype(str), prev["response"].fillna("")))

        for _, r in sample.iterrows():
            qid = str(r["ID"])
            if qid in existing and str(existing[qid]).strip():
                answer = existing[qid]
                used_cached = True
            else:
                q = r[question_col]
                prompt = template.replace("{q}", str(q))
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=80,
                )
                answer = resp.output_text.strip()
                used_cached = False

            qtext = r[question_col]
            # BLEnD evaluator expects: ID, question text, response, prompt
            prompt_text = template.replace("{q}", str(qtext))
            rows.append({
                "ID": qid,
                "Translation": qtext,
                "prompt": prompt_text,
                "response": answer,
                "cached": used_cached,
            })

            # Save every item so an interrupted run can resume.
            pd.DataFrame(rows).to_csv(out_file, index=False, encoding="utf-8")

        print(f"Finished {model} / {language}: {len(rows)} items")
