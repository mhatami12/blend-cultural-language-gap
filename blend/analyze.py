"""All statistics for the poster, for both settings, in one run.

RQ1  overall English vs local accuracy per model          -> McNemar exact, Wilson & bootstrap CIs
RQ2  does the gap depend on the cultural domain?           -> per-domain McNemar (Holm),
                                                              GEE language x domain interaction (primary),
                                                              discordant-pair homogeneity (robustness)
RQ3  between-setting comparison: does the gap differ       -> difference of gaps, bootstrap CI + permutation p
     between the two BLEnD settings? (descriptive; the settings differ in language, culture AND questions)
SENSITIVITY  everything above re-computed with the first-answer-only `strict` score,
             plus response length / list-answer rates per language.

Design
  ScoredResults     reads the scored CSVs and builds PairedScores (data access)
  StudyAnalysis     runs the tests and returns AnalysisTables (computation)
  SummaryWriter     renders summary.md (presentation)
  save_tables       writes the CSVs (persistence)

Outputs (results/comparison/): overall.csv, domains.csv, interaction.csv, rq3_country_difference.csv,
response_style.csv and summary.md (numbers formatted for the poster).

Usage:
    python -m blend.analyze
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import COUNTRIES, ENGLISH, MODEL_ORDER, STATS, Paths, StatsSettings, model_name
from .data import BlendDataset
from .formatting import format_p
from .stats import BetweenSettingComparison, PairedAnalysis, PairedScores

SCORES = {"official": "binary_score", "strict": "strict_score"}


# ------------------------------------------------------------------ data access
class ScoredResults:
    def __init__(self, paths: Paths):
        self.paths = paths

    def load(self, country: str, model: str) -> Optional[pd.DataFrame]:
        """English and local-language scores of one model, merged per valid question."""
        d = self.paths.scored(country)
        loc = COUNTRIES[country].language
        en_p, lo_p = d / f"{model}_English.csv", d / f"{model}_{loc}.csv"
        if not (en_p.exists() and lo_p.exists()):
            return None
        en, lo = pd.read_csv(en_p), pd.read_csv(lo_p)
        m = en[en.valid].merge(lo[lo.valid], on=["id", "topic"], suffixes=("_en", "_loc"))
        expected = len(BlendDataset(COUNTRIES[country], self.paths.blend_dir).valid_questions)
        if len(m) != expected:
            warnings.warn(f"{country}/{model}: {len(m)} complete pairs, expected {expected}. "
                          "Re-run inference to fill the gaps before reporting.")
        return m

    @staticmethod
    def pairs(merged: pd.DataFrame, scoring: str) -> PairedScores:
        col = SCORES[scoring]
        return PairedScores(pd.DataFrame({"id": merged.id, "topic": merged.topic,
                                          "en": merged[f"{col}_en"], "loc": merged[f"{col}_loc"]}))

    def response_counts(self) -> list[tuple[str, int, int]]:
        """(country, stored responses, evaluable responses) per setting."""
        rows = []
        for country, spec in COUNTRIES.items():
            files = list(self.paths.raw(country).glob("*.jsonl"))
            stored = sum(sum(1 for line in open(f, encoding="utf-8") if line.strip()) for f in files)
            n_valid = len(BlendDataset(spec, self.paths.blend_dir).valid_questions)
            rows.append((country, stored, n_valid * len(files)))
        return rows


# ------------------------------------------------------------------ computation
@dataclass
class AnalysisTables:
    overall: pd.DataFrame
    domains: pd.DataFrame
    interaction: pd.DataFrame
    response_style: pd.DataFrame
    rq3: pd.DataFrame
    counts: list


class StudyAnalysis:
    def __init__(self, results: ScoredResults, settings: StatsSettings = STATS):
        self.results = results
        self.settings = settings

    def run(self) -> AnalysisTables:
        overall, domains, inter, style, rq3 = [], [], [], [], []
        data: dict[tuple[str, str, str], PairedScores] = {}
        for country, spec in COUNTRIES.items():
            for model in MODEL_ORDER:
                m = self.results.load(country, model)
                if m is None:
                    print(f"skip {country}/{model}: scored files missing")
                    continue
                key = {"country": country, "model": model_name(model)}
                for scoring in SCORES:
                    scores = ScoredResults.pairs(m, scoring)
                    data[(country, model, scoring)] = scores
                    a = PairedAnalysis(scores, self.settings)
                    overall.append({**key, "scoring": scoring, **a.summary()})
                    for _, r in a.domain_table().iterrows():
                        domains.append({**key, "scoring": scoring, **r.to_dict()})
                    inter.append({**key, "scoring": scoring,
                                  **{f"gee_{k}": v for k, v in a.gee_interaction().items()},
                                  **{f"homog_{k}": v for k, v in a.discordance_homogeneity().items()}})
                style += self._response_style(m, key, spec.language)
        countries = list(COUNTRIES)
        for model in MODEL_ORDER:
            for scoring in SCORES:
                a, b = (data.get((c, model, scoring)) for c in countries)
                if a is not None and b is not None:
                    rq3.append({"model": model_name(model), "scoring": scoring,
                                "comparison": f"{countries[0]} - {countries[1]}",
                                **BetweenSettingComparison(a, b, self.settings).difference()})
        return AnalysisTables(pd.DataFrame(overall), pd.DataFrame(domains), pd.DataFrame(inter),
                              pd.DataFrame(style), pd.DataFrame(rq3), self.results.response_counts())

    @staticmethod
    def _response_style(m: pd.DataFrame, key: dict, local_language: str) -> list[dict]:
        return [{**key, "language": lang,
                 "median_words": m[f"n_words{sfx}"].median(),
                 "pct_over_5_words": 100 * (m[f"n_words{sfx}"] > 5).mean(),
                 "pct_list_answers": 100 * m[f"is_list{sfx}"].mean(),
                 "acc_official": 100 * m[f"binary_score{sfx}"].mean(),
                 "acc_strict": 100 * m[f"strict_score{sfx}"].mean()}
                for lang, sfx in [(ENGLISH, "_en"), (local_language, "_loc")]]


# ------------------------------------------------------------------ presentation
class SummaryWriter:
    """Renders AnalysisTables as the markdown summary used for the poster."""

    def __init__(self, tables: AnalysisTables, alpha: float = STATS.alpha):
        self.t = tables
        self.alpha = alpha

    def _verdict(self, p: float, gap: float | None = None) -> str:
        if p >= self.alpha:
            return "n.s."
        return "sig." if gap is None else ("local better" if gap < 0 else "English better")

    def header(self) -> list[str]:
        return ["# Results summary (auto-generated by `python -m blend.analyze`)", "",
                "Gap = English accuracy - local-language accuracy (pp); negative = local language better.", "",
                "**`official`** (BLEnD soft exact match) is the PRIMARY analysis. **`strict`** (first answer only)",
                "is a SENSITIVITY analysis: it asks whether a result depends on crediting list answers.", ""]

    def counts(self) -> list[str]:
        c = self.t.counts
        return (["## Response counts", "",
                 "| Setting | Stored responses | Evaluable (valid questions x model x language) |", "|---|---|---|"]
                + [f"| {country} | {st:,} | {ev:,} |" for country, st, ev in c]
                + [f"| **Total** | **{sum(r[1] for r in c):,}** | **{sum(r[2] for r in c):,}** |", "",
                   "Azerbaijan was queried for all 500 questions (31 are excluded by BLEnD's validity rule afterwards); "
                   "Iran only for its 456 valid questions. Use these numbers rather than '4,000 responses'.", ""])

    def sensitivity(self) -> list[str]:
        """Which conclusions hold under both scorings? 'Robust' = same significance decision
        (and, when significant, the same direction) with official and strict scoring."""
        L = ["## Sensitivity summary", "",
             "| Setting | Test | Official (primary) | Strict (sensitivity) | Robust? |", "|---|---|---|---|---|"]
        n_changed = 0
        for (c, m), g in self.t.overall.groupby(["country", "model"], sort=False):
            o, s = g[g.scoring == "official"].iloc[0], g[g.scoring == "strict"].iloc[0]
            vo, vs = self._verdict(o.mcnemar_p, o.gap_pp), self._verdict(s.mcnemar_p, s.gap_pp)
            n_changed += vo != vs
            L.append(f"| {c} · {m} | language gap (McNemar) | {o.gap_pp:+.1f} pp, {format_p(o.mcnemar_p)} ({vo}) | "
                     f"{s.gap_pp:+.1f} pp, {format_p(s.mcnemar_p)} ({vs}) | {'yes' if vo == vs else '**no**'} |")
        for (c, m), g in self.t.interaction.groupby(["country", "model"], sort=False):
            o, s = g[g.scoring == "official"].iloc[0], g[g.scoring == "strict"].iloc[0]
            vo, vs = self._verdict(o.gee_p), self._verdict(s.gee_p)
            L.append(f"| {c} · {m} | language × domain (GEE) | {format_p(o.gee_p)} ({vo}) | {format_p(s.gee_p)} ({vs}) | "
                     f"{'yes' if vo == vs else '**no**'} |")
        n_settings = self.t.overall[["country", "model"]].drop_duplicates().shape[0]
        return L + ["", f"{n_changed} of {n_settings} language effects depend on whether list answers are credited.", ""]

    def rq1_rq2(self) -> list[str]:
        L = []
        for scoring in SCORES:
            L += [f"## RQ1 overall - {scoring} scoring", ""]
            for _, r in self.t.overall[self.t.overall.scoring == scoring].iterrows():
                L.append(f"- {r.country} / {r.model} (n={r.n}): English {r.acc_en:.1f}% vs local {r.acc_loc:.1f}%, "
                         f"gap {r.gap_pp:+.1f} pp [{r.gap_lo:+.1f}, {r.gap_hi:+.1f}], McNemar {format_p(r.mcnemar_p)} "
                         f"(EN-only {r.en_only}, local-only {r.loc_only})")
            L += ["", f"## RQ2 language x domain - {scoring} scoring", ""]
            for _, r in self.t.interaction[self.t.interaction.scoring == scoring].iterrows():
                L.append(f"- {r.country} / {r.model}: GEE chi2({int(r.gee_df)}) = {r.gee_wald_chi2:.2f}, {format_p(r.gee_p)}; "
                         f"discordant-pair permutation {format_p(r.homog_p_permutation)}")
            d = self.t.domains[self.t.domains.scoring == scoring]
            big = d.reindex(d.gap_pp.abs().sort_values(ascending=False).index).head(4)
            L.append("- largest domain gaps: " + "; ".join(
                f"{r.country}/{r.model} {r.domain} {r.gap_pp:+.1f} pp (Holm {format_p(r.mcnemar_p_holm)})"
                for _, r in big.iterrows()))
            L.append("")
        return L

    def rq3(self) -> list[str]:
        rq3 = self.t.rq3
        L = ["## RQ3 between-setting comparison of the gap (Azerbaijan - Iran)", "",
             "The two BLEnD settings differ in language, culture AND question set, so this is a descriptive "
             "between-setting comparison, not a causal country/culture effect.", ""]
        for _, r in rq3.iterrows():
            L.append(f"- {r.model}, {r.scoring}: {r.diff_pp:+.1f} pp [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}], "
                     f"permutation {format_p(r.p_permutation)}")
        off = rq3[rq3.scoring == "official"]
        if len(off) and (off.p_permutation >= self.alpha).all():
            L += ["", "Interpretation (official = primary): the observed language gaps differed numerically between the two "
                      "settings, but the between-setting difference was not significant for either model."]
        for _, r in rq3[(rq3.scoring == "strict") & (rq3.p_permutation < self.alpha)].iterrows():
            L.append(f"Under strict scoring (sensitivity only) the difference for {r.model} reaches "
                     f"{format_p(r.p_permutation)}; report it separately as a robustness result, not as the main finding.")
        return L

    def render(self) -> str:
        L = self.header() + self.counts() + self.sensitivity() + self.rq1_rq2() + self.rq3()
        L += ["", "## Response style", "", self.t.response_style.round(1).to_markdown(index=False), ""]
        return "\n".join(L)


# ------------------------------------------------------------------ persistence
def save_tables(tables: AnalysisTables, paths: Paths) -> None:
    paths.comparison.mkdir(parents=True, exist_ok=True)
    for name, df in [("overall", tables.overall), ("domains", tables.domains), ("interaction", tables.interaction),
                     ("response_style", tables.response_style), ("rq3_country_difference", tables.rq3)]:
        df.to_csv(paths.comparison / f"{name}.csv", index=False, float_format="%.6f")
    for country in COUNTRIES:  # per-country copies for convenience
        out = paths.analysis(country)
        out.mkdir(parents=True, exist_ok=True)
        tables.overall[tables.overall.country == country].to_csv(out / "overall.csv", index=False, float_format="%.6f")
        tables.domains[tables.domains.country == country].to_csv(out / "domains.csv", index=False, float_format="%.6f")
    (paths.comparison / "summary.md").write_text(SummaryWriter(tables).render(), encoding="utf-8")


def main():
    paths = Paths.from_env()
    tables = StudyAnalysis(ScoredResults(paths)).run()
    save_tables(tables, paths)
    print((paths.comparison / "summary.md").read_text())


if __name__ == "__main__":
    main()
