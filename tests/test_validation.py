"""Manual-validation workflow and the reported numbers. Run: pytest -q tests/test_validation.py"""
import shutil
from pathlib import Path

import openpyxl
import pytest

from blend.validation import CountryValidation

ROOT = Path(__file__).resolve().parent.parent
AZ = ROOT / "results" / "manual" / "Azerbaijan"
IR = ROOT / "results" / "manual" / "Iran"


def set_judgments(src: Path, out: Path, labels):
    """Copy a reviewer workbook and overwrite its judgment column (None = empty)."""
    wb = openpyxl.load_workbook(src)
    ws = wb["Validation"]
    hdr_row = next(r for r in ws.iter_rows() if any(c.value == "ID" for c in r))
    col = next(c.column for c in hdr_row if c.value and "Judgment" in str(c.value))
    for i, row in enumerate(ws.iter_rows(min_row=hdr_row[0].row + 1)):
        if row[0].value:
            ws.cell(row=row[0].row, column=col).value = labels[i % len(labels)]  # value=None in cell() would not clear
    wb.save(out)


@pytest.fixture
def az_copy(tmp_path):
    d = tmp_path / "Azerbaijan"
    shutil.copytree(AZ, d)
    return d


def test_azerbaijan_two_independent_reviewers():
    res = CountryValidation("Azerbaijan", AZ).run()
    text = "\n".join(res.lines)
    assert "2 completed reviewers: reviewer_A, reviewer_B" in text and "pending" not in text
    ab = next(r for r in res.rows if r["comparison"] == "reviewer_A vs reviewer_B")
    assert round(ab["kappa_3cat"], 3) == 0.228 and round(ab["kappa_confident"], 3) == 0.290
    auto = next(r for r in res.rows if r["comparison"] == "automatic vs consensus")
    assert (auto["auto1_confirmed"], auto["auto1_n"], auto["auto0_actually_correct"], auto["auto0_n"]) == (19, 19, 3, 6)


def test_iran_numbers_unchanged():
    res = CountryValidation("Iran", IR).run()
    ab = next(r for r in res.rows if r["comparison"] == "reviewer_A vs reviewer_B")
    assert round(ab["kappa_3cat"], 3) == 0.392 and round(ab["kappa_confident"], 3) == 0.611
    auto = next(r for r in res.rows if r["comparison"] == "automatic vs consensus")
    assert (auto["auto1_confirmed"], auto["auto1_n"], auto["auto0_actually_correct"], auto["auto0_n"]) == (17, 17, 8, 11)


def test_supplementary_is_reported_but_not_mixed_in(az_copy):
    without_v2 = CountryValidation("Azerbaijan", az_copy).run()
    (az_copy / "supplementary").mkdir()
    shutil.copy(az_copy / "sample.csv", az_copy / "supplementary" / "annotator_v2.csv")  # v2 judgments live in sample.csv
    with_v2 = CountryValidation("Azerbaijan", az_copy).run()
    reviewer_rows = lambda r: [x for x in r.rows if not x["comparison"].startswith("supplementary")]  # noqa: E731
    assert reviewer_rows(with_v2) == reviewer_rows(without_v2)
    assert any("supplementary" in l and "annotator_v2" in l for l in with_v2.lines)
    assert not any("supplementary" in l for l in without_v2.lines)


def test_empty_sheet_is_pending(az_copy):
    set_judgments(AZ / "reviewers" / "reviewer_B.xlsx", az_copy / "reviewers" / "reviewer_B.xlsx", [None])
    text = "\n".join(CountryValidation("Azerbaijan", az_copy).run().lines)
    assert "pending (empty, ignored): reviewer_B.xlsx" in text and "1 completed annotator" in text


def test_superseded_reviewer_is_not_used():
    """The replaced reviewer A is kept in superseded/ for transparency but never read."""
    assert (AZ / "superseded" / "reviewer_A_old.xlsx").exists()
    text = "\n".join(CountryValidation("Azerbaijan", AZ).run().lines)
    assert "reviewer_A_old" not in text and "superseded" not in text


def test_items_matched_without_model_column():
    """The Azerbaijani sheets have no model column and two ids occur twice; id + response matching
    must still find all 50 items in sample.csv."""
    text = "\n".join(CountryValidation("Azerbaijan", AZ).run().lines)
    assert "WARNING" not in text and "50 items" in text
