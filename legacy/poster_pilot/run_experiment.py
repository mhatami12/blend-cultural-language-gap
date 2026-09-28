import csv, json, os, re, time
from pathlib import Path
from urllib.request import urlopen, Request

from openai import OpenAI

BASE = Path(__file__).resolve().parent
SELECTED = BASE / "selected_pilot_questions.csv"

IRAN_ANN = "https://raw.githubusercontent.com/nlee0212/BLEnD/main/data/annotations/Iran_data.json"
AZ_ANN   = "https://raw.githubusercontent.com/nlee0212/BLEnD/main/data/annotations/Azerbaijan_data.json"
IRAN_Q   = "https://raw.githubusercontent.com/nlee0212/BLEnD/main/data/questions/Iran_questions.csv"
AZ_Q     = "https://raw.githubusercontent.com/nlee0212/BLEnD/main/data/questions/Azerbaijan_questions.csv"

def get_text(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

def load_json(url):
    return json.loads(get_text(url))

def load_questions(url):
    rows = list(csv.DictReader(get_text(url).splitlines()))
    out = {}
    for row in rows:
        out[row["ID"]] = row
    return out

def normalize(s):
    s = s.strip().lower()
    s = re.sub(r"[\u200c\u200d]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.!?؟،,:;]+$", "", s)
    return s

def top_gold(record, english=False):
    anns = record.get("annotations", [])
    if not anns:
        return []
    answers = []
    for a in anns:
        vals = a.get("en_answers" if english else "answers", [])
        for v in vals:
            answers.append((v, int(a.get("count", 0))))
    answers.sort(key=lambda x: x[1], reverse=True)
    return answers

def simple_gold_match(pred, gold_list):
    p = normalize(pred)
    if not p:
        return False
    for g, _ in gold_list:
        gn = normalize(g)
        if not gn:
            continue
        if p == gn or gn in p or p in gn:
            return True
    return False

def prompt(question, lang):
    if lang == "English":
        instruction = (
            "Answer the question using your knowledge of everyday life and culture. "
            "Give a short direct answer in English. Do not explain unless necessary."
        )
    elif lang == "Persian":
        instruction = (
            "با توجه به دانش خود درباره زندگی روزمره و فرهنگ ایران، فقط یک پاسخ کوتاه و مستقیم به فارسی بده. "
            "توضیح اضافه نده مگر اینکه لازم باشد."
        )
    else:
        instruction = (
            "Gündəlik həyat və mədəniyyət haqqında bildiklərinizə əsaslanaraq suala qısa və birbaşa cavab verin. "
            "Cavabı Azərbaycan dilində verin və əlavə izah yazmayın."
        )
    return f"{instruction}\n\nQuestion:\n{question}"

def call_model(client, model, text):
    resp = client.responses.create(
        model=model,
        input=text,
        temperature=0,
        max_output_tokens=120,
    )
    return resp.output_text.strip()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")
    models = [m.strip() for m in [os.getenv("MODEL_A","gpt-5-mini"), os.getenv("MODEL_B","") ] if m.strip()]
    client = OpenAI(api_key=api_key)

    selected = list(csv.DictReader(SELECTED.open(encoding="utf-8-sig")))
    iran_ann = load_json(IRAN_ANN)
    az_ann = load_json(AZ_ANN)
    iran_q = load_questions(IRAN_Q)
    az_q = load_questions(AZ_Q)

    rows = []
    for item in selected:
        culture = item["culture"]
        qid = item["question_id"]
        qdb = iran_q if culture == "Iran" else az_q
        adb = iran_ann if culture == "Iran" else az_ann
        q = qdb[qid]
        rec = adb[qid]
        local_lang = "Persian" if culture == "Iran" else "Azerbaijani"
        local_question = q["Question"]
        english_question = q["Translation"]

        for model in models:
            local_answer = call_model(client, model, prompt(local_question, local_lang))
            time.sleep(0.2)
            english_answer = call_model(client, model, prompt(english_question, "English"))

            local_gold = top_gold(rec, english=False)
            en_gold = top_gold(rec, english=True)

            rows.append({
                "culture": culture,
                "question_id": qid,
                "topic": item["topic"],
                "model": model,
                "local_answer": local_answer,
                "english_answer": english_answer,
                "local_correct_simple": int(simple_gold_match(local_answer, local_gold)),
                "english_correct_simple": int(simple_gold_match(english_answer, en_gold)),
                "gold_local_top": " | ".join(g for g,_ in local_gold[:3]),
                "gold_english_top": " | ".join(g for g,_ in en_gold[:3]),
            })

    out = BASE / "raw_responses.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Summaries
    from collections import defaultdict
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for r in rows:
        groups[(r["model"], r["culture"])].append(r)
        pairs[(r["model"], r["culture"], r["question_id"])] = r

    summary_rows = []
    for (model, culture), vals in groups.items():
        n = len(vals)
        le = sum(int(v["english_correct_simple"]) for v in vals)/n*100
        ll = sum(int(v["local_correct_simple"]) for v in vals)/n*100
        both = sum(int(v["english_correct_simple"]) and int(v["local_correct_simple"]) for v in vals)/n*100
        en_only = sum(int(v["english_correct_simple"]) and not int(v["local_correct_simple"]) for v in vals)/n*100
        local_only = sum(int(v["local_correct_simple"]) and not int(v["english_correct_simple"]) for v in vals)/n*100
        summary_rows.append({
            "model": model, "culture": culture, "n": n,
            "english_accuracy_pct": round(le,1),
            "local_accuracy_pct": round(ll,1),
            "both_correct_pct": round(both,1),
            "english_correct_local_wrong_pct": round(en_only,1),
            "local_correct_english_wrong_pct": round(local_only,1),
            "english_minus_local_pp": round(le-ll,1),
        })

    with (BASE/"summary_by_model_and_language.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)

    # Paired table for poster/error analysis
    pair_rows = []
    for (model, culture, qid), r in pairs.items():
        pair_rows.append({
            "model": model, "culture": culture, "question_id": qid,
            "english_correct": r["english_correct_simple"],
            "local_correct": r["local_correct_simple"],
            "pattern": (
                "both_correct" if r["english_correct_simple"] and r["local_correct_simple"]
                else "english_only" if r["english_correct_simple"]
                else "local_only" if r["local_correct_simple"]
                else "both_wrong"
            ),
            "english_answer": r["english_answer"],
            "local_answer": r["local_answer"],
        })
    with (BASE/"paired_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=pair_rows[0].keys())
        w.writeheader(); w.writerows(pair_rows)

    print("Done. Files written to:", BASE)
    print("Use summary_by_model_and_language.csv for the main chart and paired_results.csv for error analysis.")

if __name__ == "__main__":
    main()
