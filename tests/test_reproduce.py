"""End-to-end checks: the new pipeline must reproduce the numbers on the v7 poster.

Run after `./run_all.sh` (needs results/comparison/*.csv):  pytest -q tests/test_reproduce.py
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CMP = ROOT / "results" / "comparison"
pytestmark = pytest.mark.skipif(not (CMP / "overall.csv").exists(), reason="run ./run_all.sh first")

# (country, model) -> (n, English %, local %, McNemar p) as printed on poster v7
POSTER_V7_OVERALL = {
    ("Azerbaijan", "Gemini 3.5 Flash-Lite"): (469, 56.1, 65.7, "<.001"),
    ("Azerbaijan", "GPT-4.1"): (469, 70.6, 69.7, .781),
    ("Iran", "Gemini 3.5 Flash-Lite"): (456, 57.7, 65.1, .015),
    ("Iran", "GPT-4.1"): (456, 69.7, 73.7, .127),
}
# (country, model) -> (GEE chi2, p) as printed on poster v7
POSTER_V7_GEE = {
    ("Azerbaijan", "Gemini 3.5 Flash-Lite"): (14.29, .014),
    ("Azerbaijan", "GPT-4.1"): (11.69, .039),
    ("Iran", "Gemini 3.5 Flash-Lite"): (25.33, "<.001"),
    ("Iran", "GPT-4.1"): (None, .224),
}
# domain gaps quoted on poster v7 (Gemini)
POSTER_V7_DOMAINS = {
    ("Azerbaijan", "Food"): -19.4, ("Azerbaijan", "Holidays/Celebration/Leisure"): -18.7,
    ("Iran", "Food"): -28.1, ("Iran", "Holidays/Celebration/Leisure"): -16.9,
}


def check_p(p, expected):
    if expected == "<.001":
        assert p < .001
    else:
        assert p == pytest.approx(expected, abs=0.0006)  # poster rounds to 3 decimals


@pytest.fixture(scope="module")
def overall():
    o = pd.read_csv(CMP / "overall.csv")
    return o[o.scoring == "official"].set_index(["country", "model"])


@pytest.mark.parametrize("key", POSTER_V7_OVERALL)
def test_overall_matches_poster(overall, key):
    n, en, loc, p = POSTER_V7_OVERALL[key]
    r = overall.loc[key]
    assert r.n == n
    assert round(r.acc_en, 1) == en
    assert round(r.acc_loc, 1) == loc
    check_p(r.mcnemar_p, p)


@pytest.mark.parametrize("key", POSTER_V7_GEE)
def test_gee_matches_poster(key):
    g = pd.read_csv(CMP / "interaction.csv")
    r = g[g.scoring == "official"].set_index(["country", "model"]).loc[key]
    chi2, p = POSTER_V7_GEE[key]
    if chi2 is not None:
        assert round(r.gee_wald_chi2, 2) == chi2
    check_p(r.gee_p, p)


@pytest.mark.parametrize("key", POSTER_V7_DOMAINS)
def test_domain_gaps_match_poster(key):
    d = pd.read_csv(CMP / "domains.csv")
    r = d[(d.scoring == "official") & (d.model == "Gemini 3.5 Flash-Lite")].set_index(["country", "domain"]).loc[key]
    assert round(r.gap_pp, 1) == POSTER_V7_DOMAINS[key]


def test_v7_figure_bug_is_fixed():
    """v7 figure plotted Azerbaijan / GPT-4.1 / Education at 0; the data say +7.2 pp."""
    d = pd.read_csv(CMP / "domains.csv")
    r = d[(d.scoring == "official") & (d.country == "Azerbaijan") & (d.model == "GPT-4.1") & (d.domain == "Education")]
    assert round(r.gap_pp.iloc[0], 1) == 7.2


def test_all_valid_questions_answered():
    o = pd.read_csv(CMP / "overall.csv")
    assert set(o.groupby("country").n.unique().map(tuple)) == {(469,), (456,)}


def test_strict_scoring_present():
    o = pd.read_csv(CMP / "overall.csv")
    assert set(o.scoring) == {"official", "strict"}


def test_manual_validation_matches_poster():
    """Poster v7: Persian reviewers 66.0% (kappa .392); confident subset 90.3% (kappa .611)."""
    v = pd.read_csv(ROOT / "results" / "manual" / "validation_summary.csv")
    r = v[(v.country == "Iran") & v.comparison.str.contains("reviewer_A vs reviewer_B")].iloc[0]
    assert round(r.agree_3cat, 1) == 66.0 and round(r.kappa_3cat, 3) == 0.392
    assert round(r.agree_confident, 1) == 90.3 and round(r.kappa_confident, 3) == 0.611
