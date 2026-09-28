import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_prompt(base: Path, language: str):
    p = base / "data" / "prompts" / "Iran_prompts.csv"
    df = pd.read_csv(p)
    row = df[df["id"] == "inst-4"].iloc[0]
    key = "English" if language == "English" else "Translation"
    return row[key]


def make_openai(model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def call(prompt):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128,
            seed=42,
        )
        c = r.choices[0]
        return c.message.content or "", {
            "finish_reason": c.finish_reason,
            "model_version": r.model,
        }
    return call


def make_gemini(model):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Do not send the older thinking_budget configuration.
    config = types.GenerateContentConfig(max_output_tokens=128)

    def call(prompt):
        r = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        finish = (
            r.candidates[0].finish_reason.name
            if r.candidates else "NO_CANDIDATE"
        )
        return r.text or "", {
            "finish_reason": finish,
            "model_version": getattr(r, "model_version", model),
        }
    return call


def get_caller(model):
    if model == "gpt-4.1":
        return make_openai(model)
    if model == "gemini-3.5-flash-lite":
        return make_gemini(model)
    raise ValueError(f"Unsupported model: {model}")


def call_with_retry(call, prompt, tries=6):
    last = None
    for attempt in range(tries):
        try:
            return call(prompt)
        except Exception as e:
            last = e
            wait = min(60, 2 ** attempt)
            print(f"API error: {type(e).__name__}: {str(e)[:140]}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {tries} attempts: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-dir", default="BLEnD")
    ap.add_argument("--questions", default="selected_iran_pilot_questions.csv")
    ap.add_argument("--out-dir", default="results/iran_pilot/raw")
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4.1", "gemini-3.5-flash-lite"])
    args = ap.parse_args()

    base = Path(args.blend_dir)
    q = pd.read_csv(args.questions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for language in ["English", "Persian"]:
            out = out_dir / f"{model}_{language}.jsonl"
            done = set()

            if out.exists():
                with out.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            done.add(json.loads(line)["id"])

            todo = q[~q["ID"].isin(done)]
            print(f"\n{model} / {language}: {len(done)} done, {len(todo)} remaining")

            call = get_caller(model)
            template = load_prompt(base, language)

            with out.open("a", encoding="utf-8") as f:
                for _, row in todo.iterrows():
                    question = row["Translation"] if language == "English" else row["Question"]
                    prompt = template.replace("{q}", question)

                    response, meta = call_with_retry(call, prompt)

                    rec = {
                        "id": row["ID"],
                        "topic": row["Topic"],
                        "language": language,
                        "model_key": model,
                        "api_model": model,
                        "prompt": prompt,
                        "response": response,
                        **meta,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    time.sleep(0.15)


if __name__ == "__main__":
    main()
