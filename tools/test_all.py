"""Test everything in this repository with one command.

Runs in a temporary copy, so the repository itself is never modified. No API keys needed
(the inference test uses a fake model), unless --api is given.

    ./test_all.sh                 # or: python tools/test_all.py
    ./test_all.sh --api           # also sends 1 real question per model (needs OPENAI_API_KEY / GEMINI_API_KEY)
    ./test_all.sh --keep          # keep the temporary folder for inspection

Checks
  1  environment      Python version, packages, pinned hazm 0.10.0, spaCy English model
  2  BLEnD data       folder found, pinned commit, the 6 files the code needs
  3  pipeline         deletes every derived file, re-runs score -> analyze -> validation ->
                      response_language -> manual_sample -> figures, and compares every regenerated
                      .csv/.md byte-for-byte with the delivered version
  4  pytest           unit tests + reproduction of every poster-v7 number
  5  legacy vs new    runs the OLD scorers (legacy/) on the same raw responses and compares
                      every item score with the new scorer
  6  inference        resumable inference loop with a fake model (and real APIs with --api)
  7  manual sample    validation sample is 50 rows, 25 auto-correct / 25 auto-incorrect, both countries
A report is written to test_report.md.
"""
import argparse
import filecmp
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED_COMMIT = "7b9c131719e7fe5f9bed0f8b855532d613cc9f2b"
NEEDED = [f"data/{d}/{c}_{s}" for c in ("Azerbaijan", "Iran")
          for d, s in (("questions", "questions.csv"), ("prompts", "prompts.csv"), ("annotations", "data.json"))]
DERIVED = ["results/Azerbaijan/scored", "results/Iran/scored", "results/Azerbaijan/analysis", "results/Iran/analysis",
           "results/comparison", "results/figures", "results/manual/validation_summary.csv",
           "results/manual/validation_summary.md", "results/manual/Azerbaijan/error_analysis_candidates.csv",
           "results/manual/Iran/error_analysis_candidates.csv"]
STEPS = [["blend.score"], ["blend.analyze"], ["blend.validation"], ["blend.response_language"],
         ["blend.manual_sample", "--errors-only"], ["blend.figures"]]

GREEN, RED, YELLOW, BOLD, END = ("\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m") if sys.stdout.isatty() else ("",) * 5
results, report = [], []


def record(section, name, ok, detail=""):
    status = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    color = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}[status]
    print(f"  {color}{status}{END}  {name}" + (f"  - {detail}" if detail else ""))
    results.append((section, name, status, detail))


def section(title):
    print(f"\n{BOLD}{title}{END}")
    report.append(f"\n## {title}\n")


def run(cmd, cwd, env=None, timeout=1800):
    t = time.time()
    p = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr, time.time() - t


# ------------------------------------------------------------------ 1 environment
def check_environment():
    section("1. Environment")
    record("env", f"Python {platform.python_version()}", sys.version_info >= (3, 10), "needs >= 3.10")
    for mod in ["pandas", "numpy", "scipy", "statsmodels", "matplotlib", "openpyxl", "spacy", "pytest", "tabulate"]:
        try:
            m = __import__(mod)
            record("env", f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            record("env", f"import {mod}", False, f"{e} -> run ./setup.sh")
    try:
        from importlib.metadata import version
        v = version("hazm")
        from hazm import Lemmatizer
        ok = Lemmatizer().lemmatize("کارمندی") == "کارمندی"   # hazm 0.9.x returns 'کارمند' and flips 2 Iran items
        record("env", f"hazm {v} lemmatizes like the reported runs", ok,
               "" if ok else "this hazm version changes 2 Iran items; install 0.10.0 via ./setup.sh")
    except Exception as e:  # noqa: BLE001
        record("env", "hazm", False, f"{e} -> ./setup.sh")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        record("env", "spaCy en_core_web_sm", True)
    except Exception as e:  # noqa: BLE001
        record("env", "spaCy en_core_web_sm", False, f"{e} -> python -m spacy download en_core_web_sm")
    try:
        import lingua  # noqa: F401  (availability check)
        record("env", "lingua (optional, response_language)", True)
    except ImportError:
        record("env", "lingua (optional, response_language)", None, "not installed: Azerbaijani detection less precise")


# ------------------------------------------------------------------ 2 BLEnD
def find_blend():
    cands = [os.environ.get("BLEND_DIR"), ROOT / "external" / "BLEnD", ROOT.parent / "BLEnD"]
    for c in cands:
        if c and (Path(c) / "data" / "annotations").exists():
            return Path(c).resolve()
    return None


def check_blend():
    section("2. BLEnD data")
    b = find_blend()
    if b is None:
        record("blend", "BLEnD folder", False, "not found: set BLEND_DIR=../BLEnD or run ./run_all.sh once (it clones BLEnD)")
        return None
    record("blend", "BLEnD folder", True, str(b))
    code, out, _ = run(["git", "rev-parse", "HEAD"], b)
    if code == 0:
        record("blend", "pinned commit 7b9c131", out.strip() == PINNED_COMMIT, out.strip()[:12])
    else:
        record("blend", "pinned commit 7b9c131", None, "not a git clone; cannot check the commit (data files are checked below)")
    missing = [f for f in NEEDED if not (b / f).exists()]
    record("blend", "6 required data files", not missing, ", ".join(missing))
    return b


# ------------------------------------------------------------------ 3 pipeline
def check_pipeline(tmp, blend):
    section("3. Pipeline rebuilds all results from raw responses")
    work = tmp / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", ".venv", "external", "__pycache__", ".pytest_cache",
                                                              "test_report.md"))
    for d in DERIVED:
        p = work / d
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    env = {"BLEND_DIR": str(blend), "PYTHONPATH": str(work)}
    for step in STEPS:
        code, out, secs = run([sys.executable, "-m", *step], work, env)
        record("pipeline", f"python -m {' '.join(step)}", code == 0, f"{secs:.0f}s" if code == 0 else out.strip()[-400:])
        report.append(f"<details><summary>{' '.join(step)} output</summary>\n\n```\n{out.strip()[-3000:]}\n```\n</details>\n")
        if code != 0:
            return work
    same, diff, missing, figs = [], [], [], []
    for d in DERIVED:
        orig = ROOT / d
        files = [orig] if orig.is_file() else sorted(p for p in orig.rglob("*") if p.is_file()) if orig.exists() else []
        for f in files:
            rel = f.relative_to(ROOT)
            new = work / rel
            if not new.exists():
                missing.append(str(rel))
            elif f.suffix in (".png", ".pdf"):
                figs.append((str(rel), new.stat().st_size > 1000))
            elif filecmp.cmp(f, new, shallow=False):
                same.append(str(rel))
            else:
                diff.append(str(rel))
    record("pipeline", f"{len(same)} tables/summaries identical to the delivered ones", not diff and not missing,
           ("different: " + ", ".join(diff[:6]) if diff else "") + (" missing: " + ", ".join(missing[:6]) if missing else ""))
    record("pipeline", f"{len(figs)} figure files regenerated", figs and all(ok for _, ok in figs),
           "pixels may differ from the delivered PNG/PDF with another matplotlib version; open them to look")
    if diff:
        report.append("Files that differ (compare with `diff`):\n" + "\n".join(f"- {d}" for d in diff))
    return work


# ------------------------------------------------------------------ 4 pytest
def check_pytest(work):
    section("4. pytest (unit tests + poster-v7 numbers)")
    code, out, secs = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], work,
                          {"PYTHONPATH": str(work)})
    last = [l for l in out.strip().splitlines() if l.strip()][-1] if out.strip() else ""
    record("pytest", "pytest tests/", code == 0, last)
    report.append(f"```\n{out.strip()[-4000:]}\n```")


# ------------------------------------------------------------------ 5 legacy vs new
def compare_scores(old_csv, new_csv):
    import pandas as pd
    a, b = pd.read_csv(old_csv), pd.read_csv(new_csv)
    m = a[a.valid].merge(b[b.valid], on="id", suffixes=("_old", "_new"))
    bad = m[(m.binary_score_old != m.binary_score_new) | ((m.weight_score_old - m.weight_score_new).abs() > 1e-9)]
    return len(m), bad.id.tolist()


def check_legacy(tmp, work, blend):
    section("5. Old scorers (legacy/) vs new scorer, item by item")
    # Azerbaijan: legacy/azerbaijan_src/score.py expects <root>/src, <root>/third_party, <root>/results/raw
    az = tmp / "legacy_az"
    shutil.copytree(ROOT / "legacy" / "azerbaijan_src", az / "src")
    shutil.copytree(ROOT / "third_party", az / "third_party", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "results" / "Azerbaijan" / "raw", az / "results" / "raw")
    code, out, secs = run([sys.executable, "score.py"], az / "src", {"BLEND_DIR": str(blend)})
    record("legacy", "old Azerbaijan scorer runs", code == 0, f"{secs:.0f}s" if code == 0 else out.strip()[-300:])
    if code == 0:
        for f in sorted((az / "results" / "scored").glob("*.csv")):
            n, bad = compare_scores(f, work / "results" / "Azerbaijan" / "scored" / f.name)
            record("legacy", f"Azerbaijan {f.stem}: {n} items", not bad, f"differ: {bad[:5]}" if bad else "identical")
    # Iran: legacy/iran_pilot/score_iran_pilot.py expects ./BLEnD and ./results/iran_pilot/raw
    ir = tmp / "legacy_ir"
    (ir / "results" / "iran_pilot").mkdir(parents=True)
    shutil.copy(ROOT / "legacy" / "iran_pilot" / "score_iran_pilot.py", ir)
    shutil.copytree(ROOT / "results" / "Iran" / "raw", ir / "results" / "iran_pilot" / "raw")
    try:
        (ir / "BLEnD").symlink_to(blend, target_is_directory=True)
    except OSError:
        shutil.copytree(blend / "data", ir / "BLEnD" / "data")
    code, out, secs = run([sys.executable, "score_iran_pilot.py"], ir)
    record("legacy", "old Iran scorer runs", code == 0, f"{secs:.0f}s" if code == 0 else out.strip()[-300:])
    if code == 0:
        for f in sorted((ir / "results" / "iran_pilot" / "scored").glob("*.csv")):
            n, bad = compare_scores(f, work / "results" / "Iran" / "scored" / f.name)
            record("legacy", f"Iran {f.stem}: {n} items", not bad, f"differ: {bad[:5]}" if bad else "identical")


# ------------------------------------------------------------------ 6 inference
FAKE = """
from blend.config import COUNTRIES, MODELS, GenerationSettings, Paths
from blend.data import BlendDataset
from blend.inference import Generation, InferenceRunner, ModelClient, ResponseStore

class FakeClient(ModelClient):
    def generate(self, prompt):
        return Generation("Plov, dolma", {"finish_reason": "stop", "model_version": "fake", "temperature": 0})

paths = Paths.from_env()
client = FakeClient(MODELS["gpt-4.1"], GenerationSettings())
for country, lang in [("Azerbaijan", "Azerbaijani"), ("Iran", "Persian")]:
    runner = InferenceRunner(BlendDataset(COUNTRIES[country], paths.blend_dir), client,
                             ResponseStore.for_run(paths, country, "gpt-4.1", lang))
    runner.run(lang, limit=2)
    runner.run(lang, limit=2)   # resume: must add the NEXT 2, not repeat
"""


def check_inference(tmp, work, blend, api):
    section("6. Inference")
    res = tmp / "fake_results"
    code, out, _ = run([sys.executable, "-c", FAKE], work, {"BLEND_DIR": str(blend), "RESULTS_DIR": str(res),
                                                            "PYTHONPATH": str(work)})
    record("inference", "fake-model run", code == 0, "" if code == 0 else out.strip()[-300:])
    if code == 0:
        for country, lang in [("Azerbaijan", "Azerbaijani"), ("Iran", "Persian")]:
            recs = [json.loads(l) for l in open(res / country / "raw" / f"gpt-4.1_{lang}.jsonl", encoding="utf-8")]
            ids = [r["id"] for r in recs]
            ok = len(recs) == 4 and len(set(ids)) == 4 and all("temperature" in r and "prompt" in r for r in recs)
            record("inference", f"{country}: resume + saved fields", ok, f"{len(recs)} records, {len(set(ids))} unique ids")
    if not api:
        record("inference", "real API call", None, "skipped (use --api with OPENAI_API_KEY / GEMINI_API_KEY set)")
        return
    for model, key in [("gpt-4.1", "OPENAI_API_KEY"), ("gemini-3.5-flash-lite", "GEMINI_API_KEY")]:
        if not os.environ.get(key):
            record("inference", f"real API {model}", None, f"{key} not set")
            continue
        code, out, secs = run([sys.executable, "-m", "blend.inference", "--country", "Iran", "--model", model,
                               "--language", "local", "--limit", "1"], work,
                              {"BLEND_DIR": str(blend), "RESULTS_DIR": str(tmp / "api_results"), "PYTHONPATH": str(work)})
        f = tmp / "api_results" / "Iran" / "raw" / f"{model}_Persian.jsonl"
        ok = code == 0 and f.exists() and f.stat().st_size > 0
        record("inference", f"real API {model} (1 question)", ok,
               json.loads(f.read_text(encoding="utf-8").splitlines()[0])["response"][:60] if ok else out.strip()[-300:])


# ------------------------------------------------------------------ 7 manual sample
def check_manual_sample(work, blend):
    section("7. Manual sample + error-analysis sheets")
    code, out, _ = run([sys.executable, "-m", "blend.manual_sample"], work, {"BLEND_DIR": str(blend), "PYTHONPATH": str(work)})
    record("manual", "python -m blend.manual_sample", code == 0, "" if code == 0 else out.strip()[-300:])
    if code != 0:
        return
    import pandas as pd
    for c in ("Azerbaijan", "Iran"):
        s = pd.read_csv(work / "results" / "manual" / c / "sample_new.csv")
        ok = len(s) == 50 and (s.automatic_score == 1).sum() == 25 and not s.duplicated(["id", "model"]).any()
        record("manual", f"{c}: sample_new.csv", ok, f"{len(s)} rows, {(s.automatic_score == 1).sum()} auto-correct, models: "
               + ", ".join(f"{k}={v}" for k, v in s.model.value_counts().items()))
        e = pd.read_csv(work / "results" / "manual" / c / "error_analysis_candidates.csv")
        record("manual", f"{c}: error_analysis_candidates.csv", len(e) > 0, f"{len(e)} language-dependent cases")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", action="store_true", help="also make 1 real API call per model")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = ap.parse_args()
    print(f"{BOLD}Testing {ROOT}{END}")
    check_environment()
    blend = check_blend()
    tmp = Path(tempfile.mkdtemp(prefix="blend_test_"))
    try:
        if blend and not any(s == "FAIL" for sec, _, s, _ in results if sec == "env"):
            work = check_pipeline(tmp, blend)
            check_pytest(work)
            check_legacy(tmp, work, blend)
            check_inference(tmp, work, blend, a.api)
            check_manual_sample(work, blend)
        else:
            print(f"\n{RED}Fix the failures above first (usually: ./setup.sh, and BLEND_DIR=../BLEnD).{END}")
    finally:
        if a.keep:
            print(f"\ntemporary folder kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    n = {s: sum(1 for *_, st, _ in results if st == s) for s in ("PASS", "WARN", "FAIL")}
    color = GREEN if n["FAIL"] == 0 else RED
    print(f"\n{BOLD}{color}{n['PASS']} passed, {n['FAIL']} failed, {n['WARN']} warnings{END}")
    table = ["| Section | Check | Result | Detail |", "|---|---|---|---|"] + [
        f"| {s} | {name} | {st} | {str(d).replace('|', '/')[:200]} |" for s, name, st, d in results]
    (ROOT / "test_report.md").write_text(
        f"# Test report\n\n{time.strftime('%Y-%m-%d %H:%M')} - Python {platform.python_version()} - "
        f"**{n['PASS']} passed, {n['FAIL']} failed, {n['WARN']} warnings**\n\n" + "\n".join(table) + "\n" + "\n".join(report),
        encoding="utf-8")
    print(f"report: {ROOT / 'test_report.md'}")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
