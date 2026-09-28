import os
"""Run BLEnD Azerbaijan SAQ inference with resumable checkpointing.

Every response is appended to results/raw/{model}_{language}.jsonl immediately.
Re-running the same command skips question IDs that are already saved, so an
interrupted run simply continues where it stopped.

Usage:
    python src/inference.py --model gpt-4.1 --language English
    python src/inference.py --model gpt-4.1 --language all
    python src/inference.py --model all --language all
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

from tqdm import tqdm

from common import CORE_MODELS, GEN, LANGUAGES, MODELS, RAW_DIR, build_prompt, load_prompt_templates, load_questions


# ---------------------------------------------------------------- providers
def make_caller(provider: str, api_model: str):
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY

        def call(prompt):
            r = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=GEN["temperature"],
                max_tokens=GEN["max_tokens"],
                seed=42,
            )
            c = r.choices[0]
            return c.message.content or "", {
                "finish_reason": c.finish_reason,
                "model_version": r.model,
                "system_fingerprint": r.system_fingerprint,
            }

    elif provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
        config = types.GenerateContentConfig(
            max_output_tokens=GEN["max_tokens"],
        )

        def call(prompt):
            r = client.models.generate_content(model=api_model, contents=prompt, config=config)
            finish = r.candidates[0].finish_reason.name if r.candidates else "NO_CANDIDATE"
            return (r.text or ""), {"finish_reason": finish, "model_version": r.model_version}

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

        def call(prompt):
            r = client.messages.create(
                model=api_model,
                max_tokens=GEN["max_tokens"],
                temperature=GEN["temperature"],
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in r.content if b.type == "text")
            return text, {"finish_reason": r.stop_reason, "model_version": r.model}

    else:
        raise ValueError(provider)
    return call


def with_retries(call, prompt, tries=6):
    for attempt in range(tries):
        try:
            return call(prompt)
        except Exception as e:  # rate limits, timeouts, 5xx
            wait = min(60, 2 ** attempt)
            tqdm.write(f"  error ({type(e).__name__}: {str(e)[:120]}), retry in {wait}s")
            time.sleep(wait)
    return None  # not saved -> picked up again on the next run


# ---------------------------------------------------------------- main loop
def run(model_key: str, language: str):
    provider, api_model = MODELS[model_key]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{model_key}_{language}.jsonl"

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    questions = load_questions()
    templates = load_prompt_templates()
    todo = questions[~questions["ID"].isin(done)]
    print(f"{model_key} / {language}: {len(done)} done, {len(todo)} to go -> {out_path}")
    if todo.empty:
        return

    call = make_caller(provider, api_model)
    failed = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for _, row in tqdm(todo.iterrows(), total=len(todo)):
            prompt = build_prompt(row, language, templates)
            result = with_retries(call, prompt)
            if result is None:
                failed += 1
                continue
            text, meta = result
            rec = {
                "id": row["ID"],
                "topic": row["Topic"],
                "language": language,
                "model_key": model_key,
                "api_model": api_model,
                "prompt": prompt,
                "response": text,
                **meta,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    if failed:
        print(f"  {failed} questions failed after retries; re-run the same command to retry them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"one of {list(MODELS)} or 'all' (= {CORE_MODELS})")
    ap.add_argument("--language", default="all", help="English, Azerbaijani, or all")
    args = ap.parse_args()

    models = CORE_MODELS if args.model == "all" else [args.model]
    langs = LANGUAGES if args.language == "all" else [args.language]
    for m in models:
        if m not in MODELS:
            sys.exit(f"unknown model {m}")
        for lang in langs:
            run(m, lang)
