"""Statistics and poster figures.

RQ1  overall English vs Azerbaijani accuracy per model
     -> accuracies with Wilson 95% CIs, paired bootstrap CI of the gap, McNemar exact test
RQ2  does the gap differ across BLEnD's six domains?
     -> per-domain gaps with bootstrap CIs, per-domain McNemar exact tests (Holm-corrected),
        PRIMARY test: homogeneity of discordant pairs across domains
        (chi-square on the domain x {EN-only correct, AZ-only correct} table, with a
        permutation p-value because some cells will be small). This is the paired-data
        analogue of testing a language x domain interaction.
        ROBUSTNESS: GEE logistic regression correct ~ language * domain, clustered by question.

Usage:
    python src/analyze.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from common import ANALYSIS_DIR, FIG_DIR, RESULTS, SCORED_DIR

RNG = np.random.default_rng(42)
N_BOOT = 10_000
N_PERM = 10_000
EXPECTED_N = 469
DOMAIN_ORDER = ["Food", "Holidays/Celebration/Leisure", "Sport", "Education", "Work life", "Family"]
DOMAIN_SHORT = {"Holidays/Celebration/Leisure": "Holidays/Leisure"}
COLORS = ["#0072B2", "#E69F00", "#009E73"]  # colour-blind safe


# ------------------------------------------------------------------ data
def load_pairs():
    pairs = {}
    for en_path in sorted(SCORED_DIR.glob("*_English.csv")):
        model = en_path.stem.rsplit("_", 1)[0]
        az_path = SCORED_DIR / f"{model}_Azerbaijani.csv"
        if not az_path.exists() or model.startswith("TEST"):
            continue
        en, az = pd.read_csv(en_path), pd.read_csv(az_path)
        m = en[en.valid][["id", "topic", "binary_score"]].merge(
            az[az.valid][["id", "binary_score"]], on="id", suffixes=("_en", "_az"))
        if len(m) != EXPECTED_N:
            warnings.warn(f"{model}: {len(m)} complete pairs, expected {EXPECTED_N}. "
                          "Re-run inference to fill the gaps before reporting.")
        pairs[model] = m.rename(columns={"binary_score_en": "en", "binary_score_az": "az"})
    if not pairs:
        raise SystemExit("no complete English+Azerbaijani score files in results/scored/")
    return pairs


# ------------------------------------------------------------------ stats helpers
def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return 100 * lo, 100 * hi


def boot_gap_ci(en, az):
    n = len(en)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    gaps = (en[idx].mean(1) - az[idx].mean(1)) * 100
    return np.percentile(gaps, [2.5, 97.5])


def paired_summary(df):
    en, az = df.en.to_numpy(), df.az.to_numpy()
    n = len(df)
    both = int(((en == 1) & (az == 1)).sum())
    en_only = int(((en == 1) & (az == 0)).sum())
    az_only = int(((en == 0) & (az == 1)).sum())
    neither = int(((en == 0) & (az == 0)).sum())
    p = mcnemar([[both, en_only], [az_only, neither]], exact=True).pvalue
    lo, hi = boot_gap_ci(en, az)
    en_lo, en_hi = wilson(en.sum(), n)
    az_lo, az_hi = wilson(az.sum(), n)
    return {
        "n": n,
        "acc_en": 100 * en.mean(), "acc_en_lo": en_lo, "acc_en_hi": en_hi,
        "acc_az": 100 * az.mean(), "acc_az_lo": az_lo, "acc_az_hi": az_hi,
        "gap_pp": 100 * (en.mean() - az.mean()), "gap_lo": lo, "gap_hi": hi,
        "both_correct": both, "en_only": en_only, "az_only": az_only, "both_wrong": neither,
        "mcnemar_p": p,
    }


def discordance_homogeneity(df):
    """Do EN-only vs AZ-only discordant pairs split differently across domains?"""
    d = df[df.en != df.az]
    table = (pd.crosstab(d.topic, d.en).reindex(index=DOMAIN_ORDER, columns=[0, 1]).fillna(0))
    table = table[table.sum(axis=1) > 0]
    if (table.sum(axis=0) == 0).any() or len(table) < 2:
        return {"note": "degenerate table (all discordant pairs in one direction or one domain)",
                "n_discordant": len(d)}
    chi2, p_asym, dof, expected = chi2_contingency(table.to_numpy(), correction=False)
    labels = d.en.to_numpy().astype(int)
    dom_codes = pd.Categorical(d.topic, categories=DOMAIN_ORDER).codes
    row_tot = np.bincount(dom_codes, minlength=len(DOMAIN_ORDER))
    keep = row_tot > 0
    count = 0
    for _ in range(N_PERM):
        perm = RNG.permutation(labels)
        ones = np.bincount(dom_codes, weights=perm, minlength=len(DOMAIN_ORDER))
        t = np.stack([row_tot - ones, ones], axis=1)[keep]
        exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.nansum((t - exp) ** 2 / exp)
        if stat >= chi2 - 1e-9:
            count += 1
    return {"chi2": chi2, "dof": dof, "p_asymptotic": p_asym, "p_permutation": (count + 1) / (N_PERM + 1),
            "min_expected": float(expected.min()), "n_discordant": len(d)}


def gee_interaction(df):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    long = pd.concat([
        df.assign(language="English", correct=df.en),
        df.assign(language="Azerbaijani", correct=df.az),
    ])[["id", "topic", "language", "correct"]]
    try:
        res = smf.gee("correct ~ C(language, Treatment('English')) * C(topic)", groups="id", data=long,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        names = [n for n in res.params.index if ":" in n]
        R = np.zeros((len(names), len(res.params)))
        for i, nm in enumerate(names):
            R[i, list(res.params.index).index(nm)] = 1
        w = res.wald_test(R, scalar=True)
        return {"wald_chi2": float(w.statistic), "df": len(names), "p": float(w.pvalue)}
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------ figures
def setup_style():
    plt.rcParams.update({"font.size": 22, "axes.titlesize": 24, "axes.labelsize": 22,
                         "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")  # vector, for the poster
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_overall(overall):
    fig, ax = plt.subplots(figsize=(10, 7))
    models = overall.index.tolist()
    x = np.arange(len(models))
    w = 0.36
    for j, (lang, col) in enumerate([("en", "English"), ("az", "Azerbaijani")]):
        vals = overall[f"acc_{lang}"]
        err = [vals - overall[f"acc_{lang}_lo"], overall[f"acc_{lang}_hi"] - vals]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, yerr=err, capsize=6, label=col,
                      color=["#0072B2", "#D55E00"][j])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, 2, f"{v:.1f}", ha="center", color="white", fontweight="bold")
    ax.set_xticks(x, models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Same questions, different language (n = {int(overall.n.iloc[0])})")
    save(fig, "fig1_overall_accuracy")


def fig_paired(overall):
    fig, ax = plt.subplots(figsize=(12, 1.6 + 1.3 * len(overall)))
    parts = [("both_correct", "Both correct", "#009E73"), ("en_only", "English only", "#0072B2"),
             ("az_only", "Azerbaijani only", "#D55E00"), ("both_wrong", "Both wrong", "#BBBBBB")]
    left = np.zeros(len(overall))
    for col, lab, c in parts:
        v = 100 * overall[col] / overall.n
        ax.barh(overall.index, v, left=left, color=c, label=lab)
        for i, (l, vv) in enumerate(zip(left, v)):
            if vv >= 5:
                ax.text(l + vv / 2, i, f"{vv:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        left += v
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of paired questions (%)")
    ax.invert_yaxis()
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=18)
    save(fig, "fig3_paired_outcomes")


def fig_domain(domain):
    models = domain.model.unique().tolist()
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(DOMAIN_ORDER))
    off = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else [0]
    for k, m in enumerate(models):
        d = domain[domain.model == m].set_index("domain").reindex(DOMAIN_ORDER)
        ax.errorbar(d.gap_pp, y + off[k], xerr=[d.gap_pp - d.gap_lo, d.gap_hi - d.gap_pp], fmt="o",
                    ms=12, capsize=5, lw=2.5, color=COLORS[k % 3], label=m)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_yticks(y, [f"{DOMAIN_SHORT.get(d, d)} (n={int(domain[domain.domain == d].n.iloc[0])})"
                      for d in DOMAIN_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("English − Azerbaijani accuracy (pp), 95% CI")
    ax.legend(frameon=False, loc="best")
    save(fig, "fig2_domain_gap")


# ------------------------------------------------------------------ manual validation
def manual_validation():
    path = RESULTS / "manual" / "manual_validation.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    m = m[m.human_correct.isin([0, 1])]
    if m.empty:
        return None
    tab = pd.crosstab(m.auto_correct, m.human_correct, rownames=["automatic"], colnames=["human"])
    agree = 100 * (m.auto_correct == m.human_correct).mean()
    return tab, agree, len(m)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    pairs = load_pairs()

    overall = pd.DataFrame({m: paired_summary(df) for m, df in pairs.items()}).T
    overall.to_csv(ANALYSIS_DIR / "rq1_overall.csv", float_format="%.4f")

    rows, homog, gee = [], {}, {}
    for m, df in pairs.items():
        drows = []
        for dom in DOMAIN_ORDER:
            s = paired_summary(df[df.topic == dom])
            drows.append({"model": m, "domain": dom, **s})
        pvals = [r["mcnemar_p"] for r in drows]
        adj = multipletests(pvals, method="holm")[1]
        for r, a in zip(drows, adj):
            r["mcnemar_p_holm"] = a
        rows += drows
        homog[m] = discordance_homogeneity(df)
        gee[m] = gee_interaction(df)
    domain = pd.DataFrame(rows)
    domain.to_csv(ANALYSIS_DIR / "rq2_domains.csv", index=False, float_format="%.4f")
    pd.DataFrame(homog).T.to_csv(ANALYSIS_DIR / "rq2_homogeneity_test.csv", float_format="%.4f")
    pd.DataFrame(gee).T.to_csv(ANALYSIS_DIR / "rq2_gee_interaction.csv", float_format="%.4f")

    fig_overall(overall)
    fig_paired(overall)
    fig_domain(domain)

    pd.set_option("display.width", 200)
    print("=== RQ1 overall ===")
    print(overall[["n", "acc_en", "acc_az", "gap_pp", "gap_lo", "gap_hi", "en_only", "az_only", "mcnemar_p"]]
          .astype(float).round(3))
    print("\n=== RQ2 per domain ===")
    print(domain[["model", "domain", "n", "acc_en", "acc_az", "gap_pp", "gap_lo", "gap_hi",
                  "en_only", "az_only", "mcnemar_p_holm"]].round(3).to_string(index=False))
    print("\n=== RQ2 primary test: homogeneity of discordant pairs across domains ===")
    print(pd.DataFrame(homog).T.round(4))
    print("\n=== RQ2 robustness: GEE language x domain interaction ===")
    print(pd.DataFrame(gee).T)

    mv = manual_validation()
    if mv:
        tab, agree, n = mv
        print(f"\n=== Manual validation (n={n}) ===\n{tab}\nagreement = {agree:.1f}%")
    print(f"\nTables -> {ANALYSIS_DIR}\nFigures -> {FIG_DIR}")
