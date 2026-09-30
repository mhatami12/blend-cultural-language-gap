"""Check every number printed on the poster against an independent recomputation.

Everything except the Persian audit is recomputed from the scored files and reviewer sheets
with plain pandas / scipy / statsmodels, without the blend.stats or blend.analyze code.
Run from the repository root:

    python tools/verify_poster_numbers.py            # BLEnD not needed
    BLEND_DIR=external/BLEnD python tools/verify_poster_numbers.py   # also checks the Persian audit
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SETTINGS = [("Azerbaijan", "Azerbaijani", "AZ"), ("Iran", "Persian", "IR")]
MODELS = [("gpt-4.1", "GPT-4.1"), ("gemini-3.5-flash-lite", "Gemini")]
DOMAINS = ["Food", "Holidays/Celebration/Leisure", "Sport", "Education", "Work life", "Family"]
rows = []


def check(section: str, what: str, poster: str, computed: str) -> None:
    rows.append((section, what, poster, computed, "OK" if poster == computed else "MISMATCH"))


def fmt_p(p: float) -> str:
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0")


def scored(country: str, model: str, lang: str) -> pd.DataFrame:
    d = pd.read_csv(RES / country / "scored" / f"{model}_{lang}.csv")
    return d[d.valid]


def paired(country: str, local: str, model: str, col: str = "binary_score") -> pd.DataFrame:
    en, loc = scored(country, model, "English"), scored(country, model, local)
    m = en[["id", "topic", col]].merge(loc[["id", col]], on="id", suffixes=("_en", "_loc"))
    return m.rename(columns={f"{col}_en": "en", f"{col}_loc": "lc"})  # not "loc": that is a pandas attribute


def mcnemar(m: pd.DataFrame) -> tuple[int, int, float]:
    b = int(((m.en == 1) & (m.lc == 0)).sum())
    c = int(((m.en == 0) & (m.lc == 1)).sum())
    return b, c, binomtest(b, b + c, 0.5).pvalue


# ------------------------------------------------------------------ poster values (from poster.tex)
POSTER = {
    "n": {"AZ": "469", "IR": "456"}, "evaluable": "3,700",
    "acc": {("AZ", "GPT-4.1"): ("70.6", "69.7"), ("AZ", "Gemini"): ("56.1", "65.7"),
            ("IR", "GPT-4.1"): ("69.7", "73.7"), ("IR", "Gemini"): ("57.7", "65.1")},
    "gap_p": {("AZ", "GPT-4.1"): ("+0.9", ".781"), ("AZ", "Gemini"): ("-9.6", "<.001"),
              ("IR", "GPT-4.1"): ("-3.9", ".127"), ("IR", "Gemini"): ("-7.5", ".015")},
    "strict": {("AZ", "GPT-4.1"): ("-6.2", ".014"), ("AZ", "Gemini"): ("-9.8", "<.001"),
               ("IR", "GPT-4.1"): ("-7.7", ".005"), ("IR", "Gemini"): ("-2.0", ".550")},
    "gee_p": {("AZ", "GPT-4.1"): ".039", ("AZ", "Gemini"): ".014", ("IR", "GPT-4.1"): ".224", ("IR", "Gemini"): "<.001"},
    "food": {("AZ", "Gemini"): "-19.4", ("IR", "Gemini"): "-28.1"},
    "filled": "AZ/Gemini/Food, AZ/Gemini/Holidays/Celebration/Leisure, IR/Gemini/Food",
    "rq3": {"GPT-4.1": ("+4.8", ".156"), "Gemini": ("-2.1", ".606")},
    "paired": {("AZ", "GPT-4.1"): "58/13/12/17", ("AZ", "Gemini"): "48/8/17/26",
               ("IR", "GPT-4.1"): "58/12/16/15", ("IR", "Gemini"): "41/16/24/18"},
    "validation": {"AZ": ("75.8", ".29", "19/19", "3/6"), "IR": ("90.3", ".61", "17/17", "8/11")},
    "persian": {"GPT-4.1": "82.5", "Gemini": "77.9"},
}

# ------------------------------------------------------------------ counts, RQ1, strict, paired outcomes
total = 0
for country, local, cc in SETTINGS:
    for model, name in MODELS:
        m = paired(country, local, model)
        total += 2 * len(m)
        if name == "GPT-4.1":
            check("Methodology", f"{country}: valid question pairs", POSTER["n"][cc], str(len(m)))
        a_en, a_loc = 100 * m.en.mean(), 100 * m.lc.mean()
        pe, pl = POSTER["acc"][(cc, name)]
        check("RQ1 figure", f"{cc}/{name} English accuracy", pe, f"{a_en:.1f}")
        check("RQ1 figure", f"{cc}/{name} local accuracy", pl, f"{a_loc:.1f}")
        b, c, p = mcnemar(m)
        g, pp = POSTER["gap_p"][(cc, name)]
        check("RQ1 + robustness", f"{cc}/{name} gap (official)", g, f"{a_en - a_loc:+.1f}")
        check("RQ1 + robustness", f"{cc}/{name} McNemar p (official)", pp, fmt_p(p))
        # 95% Wilson CIs drawn in the RQ1 figure, compared with the file the figure is drawn from
        o = pd.read_csv(RES / "comparison" / "overall.csv")
        row = o[(o.country == country) & (o.model.str.startswith(name)) & (o.scoring == "official")].iloc[0]
        for lang, k in (("en", m.en.sum()), ("loc", m.lc.sum())):
            lo, hi = proportion_confint(k, len(m), method="wilson")
            check("RQ1 figure", f"{cc}/{name} {lang} 95% CI", f"{row[f'acc_{lang}_lo']:.1f}-{row[f'acc_{lang}_hi']:.1f}",
                  f"{100*lo:.1f}-{100*hi:.1f}")
        # strict scoring
        s = paired(country, local, model, "strict_score")
        _, _, ps = mcnemar(s)
        g, pp = POSTER["strict"][(cc, name)]
        check("Robustness", f"{cc}/{name} gap (strict)", g, f"{100*(s.en.mean()-s.lc.mean()):+.1f}")
        check("Robustness", f"{cc}/{name} McNemar p (strict)", pp, fmt_p(ps))
        # paired outcomes
        parts = [((m.en == 1) & (m.lc == 1)).sum(), b, c, ((m.en == 0) & (m.lc == 0)).sum()]
        check("Paired outcomes figure", f"{cc}/{name} both/EN only/local only/both wrong (%)",
              POSTER["paired"][(cc, name)], "/".join(f"{100*x/len(m):.0f}" for x in parts))
check("Methodology", "evaluable responses", POSTER["evaluable"], f"{total:,}")

# ------------------------------------------------------------------ RQ2: domain gaps, Holm, GEE
import statsmodels.api as sm
import statsmodels.formula.api as smf

filled, gee_sig = [], 0
for country, local, cc in SETTINGS:
    for model, name in MODELS:
        m = paired(country, local, model)
        pvals, gaps = [], {}
        for dom in DOMAINS:
            dm = m[m.topic == dom]
            gaps[dom] = 100 * (dm.en.mean() - dm.lc.mean())
            pvals.append(mcnemar(dm)[2])
        holm = multipletests(pvals, method="holm")[1]
        filled += [f"{cc}/{name}/{dom}" for dom, ph in zip(DOMAINS, holm) if ph < .05]
        if (cc, name) in POSTER["food"]:
            check("RQ2 text", f"{cc}/{name} Food gap", POSTER["food"][(cc, name)], f"{gaps['Food']:.1f}")
        long = pd.concat([m.assign(language="English", correct=m.en), m.assign(language="Local", correct=m.lc)])
        res = smf.gee("correct ~ C(language) * C(topic)", groups="id", data=long,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        inter = [i for i, n in enumerate(res.params.index) if ":" in n]
        R = np.zeros((len(inter), len(res.params)))
        R[range(len(inter)), inter] = 1
        p = float(res.wald_test(R, scalar=True).pvalue)
        gee_sig += p < .05
        check("RQ2 text", f"{cc}/{name} GEE language x domain p", POSTER["gee_p"][(cc, name)], fmt_p(p))
check("RQ2 figure", "filled (Holm-significant) markers", POSTER["filled"], ", ".join(filled))
check("RQ2 text", "'H2 supported in three of four cases'", "3", str(gee_sig))

# ------------------------------------------------------------------ RQ3: independent permutation test (other seed)
for model, name in MODELS:
    d = [paired(c, l, model) for c, l, _ in SETTINGS]
    da, db = [(x.en - x.lc).to_numpy(float) for x in d]
    obs = 100 * (da.mean() - db.mean())
    rng, pooled, n_perm, count = np.random.default_rng(2026), np.concatenate([da, db]), 20000, 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(100 * (pooled[:len(da)].mean() - pooled[len(da):].mean())) >= abs(obs) - 1e-9
    p = (count + 1) / (n_perm + 1)
    g, pp = POSTER["rq3"][name]
    check("RQ3 text", f"{name} gap difference AZ - IR", g, f"{obs:+.1f}")
    # a permutation p-value is random: accept the poster value if it lies within 3 Monte-Carlo SE
    se = np.sqrt(p * (1 - p) / n_perm)
    check("RQ3 text", f"{name} permutation p (new seed, 20k; poster within 3 SE?)", pp,
          pp if abs(float(pp) - p) <= 3 * se else f"{p:.3f}")

# ------------------------------------------------------------------ manual validation, straight from the Excel sheets
def sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=None, header=None)
    s = next(v for v in raw.values() if v.astype(str).isin(["ID", "id"]).any().any())
    h = s.index[s.astype(str).isin(["ID", "id"]).any(axis=1)][0]
    s = s.iloc[h + 1:].set_axis([str(c).strip() for c in s.iloc[h]], axis=1).dropna(how="all")
    col = lambda *k: next(c for c in s.columns if any(x in c.lower() for x in k))  # noqa: E731
    lab = s[col("judg", "human")].fillna("").astype(str).str.strip().str.lower().map(
        {"correct": "C", "incorrect": "I"}).fillna("U")
    key = s[col("id")].astype(str).str.strip() + "|" + s[col("response")].fillna("").astype(str).str.strip().str.lower()
    return pd.Series(lab.values, index=key.values)


def kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe)


for country, _, cc in SETTINGS:
    f = RES / "manual" / country
    A, B = sheet(f / "reviewers" / "reviewer_A.xlsx"), sheet(f / "reviewers" / "reviewer_B.xlsx")
    B = B.reindex(A.index)
    conf = (A != "U") & (B != "U")
    agree, k = 100 * (A[conf] == B[conf]).mean(), kappa(A[conf], B[conf])
    smp = pd.read_csv(f / "sample.csv", encoding="utf-8-sig")
    idc = next(c for c in smp.columns if c.lower() == "id")
    rc = next(c for c in smp.columns if "response" in c.lower())
    ac = next(c for c in smp.columns if "auto" in c.lower())
    auto = pd.Series(smp[ac].astype(float).astype(int).values,
                     index=(smp[idc].astype(str).str.strip() + "|" + smp[rc].fillna("").astype(str).str.strip().str.lower()).values)
    auto = auto.reindex(A.index)
    unan = conf & (A == B)
    a1, a0 = A[unan & (auto == 1)], A[unan & (auto == 0)]
    pa, pk, p1, p0 = POSTER["validation"][cc]
    check("Scorer validation", f"{cc} agreement (confident items, %)", pa, f"{agree:.1f}")
    check("Scorer validation", f"{cc} Cohen's kappa (confident items)", pk, f"{k:.2f}".lstrip("0"))
    check("Scorer validation", f"{cc} auto-correct confirmed", p1, f"{(a1 == 'C').sum()}/{len(a1)}")
    check("Scorer validation", f"{cc} auto-wrong judged correct (missed)", p0, f"{(a0 == 'C').sum()}/{len(a0)}")

# ------------------------------------------------------------------ Persian audit (needs BLEnD)
if os.environ.get("BLEND_DIR"):
    sys.path.insert(0, str(ROOT))
    from blend.config import COUNTRIES, Paths
    from blend.data import BlendDataset
    from blend.scoring import BlendScorer
    ds, sc = BlendDataset(COUNTRIES["Iran"], Paths.from_env().blend_dir), BlendScorer.default()
    T = str.maketrans({**{chr(0x6F0 + i): str(i) for i in range(10)}, **{chr(0x660 + i): str(i) for i in range(10)},
                       "‌": " ", "ي": "ی", "ك": "ک"})
    for model, name in MODELS:
        d = scored("Iran", model, "Persian")
        hits = []
        for qid, r in zip(d.id, d.response_clean):
            a = copy.deepcopy(ds.annotation(qid))
            for g in a["annotations"]:
                g["answers"] = [x.translate(T) for x in g["answers"]]
            hits.append(sc.score(a, r.translate(T) if isinstance(r, str) else r, "Persian").binary)
        check("Robustness", f"IR/{name} Persian accuracy after digit/ZWNJ normalisation", POSTER["persian"][name],
              f"{100*np.mean(hits):.1f}")
else:
    rows.append(("Robustness", "Persian audit", "82.5 / 77.9", "skipped (set BLEND_DIR)", "SKIPPED"))

# ------------------------------------------------------------------ report
out = pd.DataFrame(rows, columns=["section", "number", "poster", "recomputed", "status"])
pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_rows", 200)
print(out.to_string(index=False))
bad = (out.status == "MISMATCH").sum()
print(f"\n{(out.status == 'OK').sum()} OK, {bad} mismatch, {(out.status == 'SKIPPED').sum()} skipped")
sys.exit(1 if bad else 0)
