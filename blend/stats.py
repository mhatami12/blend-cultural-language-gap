"""Paired statistics for English vs local-language correctness.

Sign convention used EVERYWHERE (tables, figures, poster):
    gap = English accuracy - local-language accuracy   (percentage points)
    negative gap  =>  the local language is better.

Design
  PairedScores              validated value object: one row per question, 0/1 in each language
  PairedAnalysis            RQ1 + RQ2 tests on one PairedScores
  BetweenSettingComparison  RQ3: difference of gaps between two independent settings
Every random procedure uses its own seeded generator, so results do not depend on call order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from .config import DOMAIN_ORDER, STATS, StatsSettings

REQUIRED = ("id", "topic", "en", "loc")


@dataclass(frozen=True)
class PairedScores:
    """Correctness of the same questions asked in English (`en`) and the local language (`loc`)."""
    df: pd.DataFrame

    def __post_init__(self):
        missing = [c for c in REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"PairedScores needs columns {REQUIRED}, missing {missing}")

    @property
    def en(self) -> np.ndarray:
        return self.df.en.to_numpy(int)

    @property
    def loc(self) -> np.ndarray:
        return self.df["loc"].to_numpy(int)

    def __len__(self) -> int:
        return len(self.df)

    def domain(self, name: str) -> "PairedScores":
        return PairedScores(self.df[self.df.topic == name])

    def domains(self) -> list[str]:
        return [d for d in DOMAIN_ORDER if (self.df.topic == d).any()]

    def differences(self) -> np.ndarray:
        """Per-question difference en - loc (1, 0 or -1)."""
        return (self.df.en - self.df["loc"]).to_numpy(float)


def wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return 100 * lo, 100 * hi


class PairedAnalysis:
    def __init__(self, scores: PairedScores, settings: StatsSettings = STATS):
        self.scores = scores
        self.settings = settings

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.settings.seed)

    # ---------------------------------------------------------------- RQ1
    def bootstrap_gap_ci(self) -> tuple[float, float]:
        """Paired bootstrap 95% CI of the gap."""
        en, loc = self.scores.en, self.scores.loc
        idx = self._rng().integers(0, len(en), size=(self.settings.n_boot, len(en)))
        gaps = (en[idx].mean(1) - loc[idx].mean(1)) * 100
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        return lo, hi

    def summary(self) -> dict:
        """Accuracies with Wilson CIs, gap with bootstrap CI, 2x2 table, exact McNemar p."""
        en, loc, n = self.scores.en, self.scores.loc, len(self.scores)
        both = int(((en == 1) & (loc == 1)).sum())
        en_only = int(((en == 1) & (loc == 0)).sum())
        loc_only = int(((en == 0) & (loc == 1)).sum())
        neither = int(((en == 0) & (loc == 0)).sum())
        p = mcnemar([[both, en_only], [loc_only, neither]], exact=True).pvalue
        lo, hi = self.bootstrap_gap_ci()
        en_lo, en_hi = wilson(en.sum(), n)
        l_lo, l_hi = wilson(loc.sum(), n)
        return {
            "n": n,
            "acc_en": 100 * en.mean(), "acc_en_lo": en_lo, "acc_en_hi": en_hi,
            "acc_loc": 100 * loc.mean(), "acc_loc_lo": l_lo, "acc_loc_hi": l_hi,
            "gap_pp": 100 * (en.mean() - loc.mean()), "gap_lo": lo, "gap_hi": hi,
            "both_correct": both, "en_only": en_only, "loc_only": loc_only, "both_wrong": neither,
            "mcnemar_p": p,
        }

    # ---------------------------------------------------------------- RQ2
    def domain_table(self) -> pd.DataFrame:
        """Per-domain summary with Holm-corrected McNemar p-values."""
        rows = [{"domain": d, **PairedAnalysis(self.scores.domain(d), self.settings).summary()}
                for d in self.scores.domains()]
        out = pd.DataFrame(rows)
        out["mcnemar_p_holm"] = multipletests(out.mcnemar_p, method="holm")[1]
        return out

    def gee_interaction(self) -> dict:
        """Language x domain interaction: GEE logistic regression with questions as clusters;
        Wald test of all interaction terms jointly (df = n_domains - 1)."""
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        df = self.scores.df
        long = pd.concat([df.assign(language="English", correct=df.en),
                          df.assign(language="Local", correct=df["loc"])])[["id", "topic", "language", "correct"]]
        long["topic"] = long.topic.astype(str)
        res = smf.gee("correct ~ C(language, Treatment('English')) * C(topic)", groups="id", data=long,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        names = [n for n in res.params.index if ":" in n]
        R = np.zeros((len(names), len(res.params)))
        for i, nm in enumerate(names):
            R[i, list(res.params.index).index(nm)] = 1
        w = res.wald_test(R, scalar=True)
        return {"wald_chi2": float(w.statistic), "df": len(names), "p": float(w.pvalue)}

    def discordance_homogeneity(self) -> dict:
        """Do EN-only vs local-only discordant pairs split differently across domains?
        Chi-square on the domain x direction table, with a permutation p-value."""
        d = self.scores.df[self.scores.df.en != self.scores.df["loc"]]
        doms = [x for x in DOMAIN_ORDER if (d.topic == x).any()]
        table = pd.crosstab(pd.Categorical(d.topic, doms), d.en).reindex(columns=[0, 1], fill_value=0)
        if (table.sum(axis=0) == 0).any() or len(table) < 2:
            return {"note": "degenerate table", "n_discordant": len(d)}
        chi2, p_asym, dof, expected = chi2_contingency(table.to_numpy(), correction=False)
        rng = self._rng()
        labels = d.en.to_numpy(int)
        codes = pd.Categorical(d.topic, doms).codes
        row_tot = np.bincount(codes, minlength=len(doms))
        count = 0
        for _ in range(self.settings.n_perm):
            ones = np.bincount(codes, weights=rng.permutation(labels), minlength=len(doms))
            t = np.stack([row_tot - ones, ones], axis=1)
            exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = np.nansum((t - exp) ** 2 / exp)
            count += stat >= chi2 - 1e-9
        return {"chi2": chi2, "dof": dof, "p_asymptotic": p_asym,
                "p_permutation": (count + 1) / (self.settings.n_perm + 1),
                "min_expected": float(expected.min()), "n_discordant": len(d)}


class BetweenSettingComparison:
    """RQ3: does the language gap differ between two BLEnD settings? The settings have
    independent question sets and differ in language, culture AND questions, so this is a
    descriptive between-setting comparison, not a causal country/culture effect."""

    def __init__(self, a: PairedScores, b: PairedScores, settings: StatsSettings = STATS):
        self.a, self.b, self.settings = a, b, settings

    def difference(self) -> dict:
        """Difference of gaps (a - b): bootstrap CI and two-sided permutation p-value."""
        da, db = self.a.differences(), self.b.differences()
        obs = 100 * (da.mean() - db.mean())
        rng = np.random.default_rng(self.settings.seed)
        boots = [100 * (rng.choice(da, len(da)).mean() - rng.choice(db, len(db)).mean())
                 for _ in range(self.settings.n_boot)]
        pooled = np.concatenate([da, db])
        count = 0
        for _ in range(self.settings.n_perm):
            rng.shuffle(pooled)
            count += abs(100 * (pooled[:len(da)].mean() - pooled[len(da):].mean())) >= abs(obs) - 1e-9
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"diff_pp": obs, "ci_lo": lo, "ci_hi": hi, "p_permutation": (count + 1) / (self.settings.n_perm + 1)}
