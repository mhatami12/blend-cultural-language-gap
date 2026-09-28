"""Manual validation: automatic score vs human judgment, and agreement between reviewers.

Layout (one folder per country):
    results/manual/<Country>/sample.csv            the sampled items with the automatic score
    results/manual/<Country>/reviewers/*.xlsx|csv  one file per reviewer / annotator

Current state:
    Iran        reviewers/reviewer_A.xlsx, reviewer_B.xlsx       two independent reviewers (complete)
    Azerbaijan  reviewers/reviewer_A.xlsx, reviewer_B.xlsx       two independent reviewers (complete)
                sample.csv also holds the earlier single-annotator pass (v2, column human_correct);
                it is not used in any statistic
Optional: files in supplementary/ are compared with the reviewers and reported separately,
never mixed into the reviewer statistics.
Empty reviewer sheets are reported as *pending* and ignored.

Column names are matched loosely, so the existing sheets work as they are:
    id        "ID" / "id"
    response  "LLM Response" / "response_raw"   (items are matched on id + response text, because
                                                  the Azerbaijani reviewer sheets have no model column and
                                                  two ids occur once per model)
    judgment  "Human Judgment" / "Reviewer Judgment" / "human_correct"
              Correct / Incorrect / Uncertain (or 1 / 0); empty = not judged yet
    auto      "Automatic Score" / "auto_correct"  (sample.csv only)

Reported per country:
  * automatic score vs human judgment, on the items where the human judgment is confident
    (Correct/Incorrect) and, when there are several reviewers, unanimous: how often an automatic
    "correct" is confirmed, and how often an automatic "incorrect" was judged correct (missed)
  * with >= 2 complete reviewers: raw agreement and Cohen's kappa on all 3 categories and on the
    items where both were confident

Design
  SheetReader        reads csv/xlsx sheets and builds the item key (id + response)
  JudgmentSheet      one reviewer's judgments (knows whether it is pending/partial)
  CountryValidation  agreement statistics for one setting
  ValidationReport   runs all settings and writes validation_summary.md/.csv

Usage:
    python -m blend.validation
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import COUNTRIES, Paths

UNCERTAIN = "Uncertain"


class SheetReader:
    """Reads review sheets (csv or xlsx) with loosely matched column names."""
    EXTENSIONS = (".xlsx", ".csv")

    @staticmethod
    def column(df: pd.DataFrame, *needles: str) -> str:
        for c in df.columns:
            if any(n in str(c).lower() for n in needles):
                return c
        raise KeyError(f"no column containing {needles} in {list(df.columns)}")

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        if path.suffix in (".xlsx", ".xls"):
            raw = pd.read_excel(path, sheet_name=None, header=None)
            has_id = lambda s: s.apply(lambda r: r.astype(str).str.strip().isin(["ID", "id"]).any(), axis=1)  # noqa: E731
            sheet = next((s for s in raw.values() if has_id(s).any()), next(iter(raw.values())))
            hdr = sheet.index[has_id(sheet)][0]
            df = sheet.iloc[hdr + 1:].copy()
            df.columns = [str(c).strip() for c in sheet.iloc[hdr]]
            return df.dropna(how="all")
        return pd.read_csv(path, encoding="utf-8-sig")

    @classmethod
    def item_key(cls, df: pd.DataFrame) -> pd.MultiIndex:
        """Item key = id + normalised response text (robust to sheets without a model column).
        fillna before astype(str): pandas 3 keeps NaN in astype(str)."""
        ids = df[cls.column(df, "id")].fillna("").astype(str).str.strip()
        resp = df[cls.column(df, "response")].fillna("").astype(str).str.strip().str.lower()
        return pd.MultiIndex.from_arrays([ids, resp], names=["id", "response"])


@dataclass
class JudgmentSheet:
    name: str
    labels: pd.Series   # Correct / Incorrect / Uncertain per item
    n_judged: int       # items with a non-empty judgment

    LABELS = {"correct": "Correct", "1": "Correct", "1.0": "Correct", "incorrect": "Incorrect", "0": "Incorrect",
              "0.0": "Incorrect", "uncertain": UNCERTAIN}
    EMPTY = {"", "nan", "none"}

    @classmethod
    def load(cls, path: Path) -> "JudgmentSheet":
        df = SheetReader.read(path)
        raw = df[SheetReader.column(df, "judg", "human")].fillna("").astype(str).str.strip().str.lower()
        labels = raw.map(lambda v: UNCERTAIN if v in cls.EMPTY else cls.LABELS.get(v, UNCERTAIN))
        return cls(path.stem, pd.Series(labels.values, index=SheetReader.item_key(df), name=path.stem),
                   int((~raw.isin(cls.EMPTY)).sum()))

    @property
    def is_pending(self) -> bool:
        return self.n_judged == 0

    @property
    def is_partial(self) -> bool:
        return 0 < self.n_judged < len(self.labels)


def load_automatic_scores(path: Path) -> pd.Series:
    df = SheetReader.read(path)
    return pd.Series(df[SheetReader.column(df, "automatic", "auto_correct")].astype(float).astype(int).values,
                     index=SheetReader.item_key(df), name="auto")


def cohen_kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """-> (observed agreement, Cohen's kappa, n)."""
    n = len(a)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan"), n


@dataclass
class ValidationResult:
    rows: list = field(default_factory=list)
    lines: list = field(default_factory=list)


class CountryValidation:
    """Manual-validation statistics for one setting (folder with sample.csv and reviewers/)."""

    def __init__(self, country: str, folder: Path):
        self.country = country
        self.folder = Path(folder)

    def _sheets_in(self, sub: str) -> list[Path]:
        return sorted(p for p in (self.folder / sub).glob("*")
                      if p.suffix in SheetReader.EXTENSIONS and not p.name.startswith("."))

    def reviewer_files(self) -> list[Path]:
        """Independent reviewers: the basis of all reported validation numbers."""
        return self._sheets_in("reviewers")

    def supplementary_files(self) -> list[Path]:
        """Other annotations (e.g. an earlier single-annotator pass), reported separately only."""
        return self._sheets_in("supplementary")

    def run(self) -> ValidationResult:
        res = ValidationResult()
        files = self.reviewer_files()
        if not files:
            res.lines.append(f"## {self.country}: no reviewer files in {self.folder / 'reviewers'}")
            return res
        sheets = [JudgmentSheet.load(p) for p in files]
        pending = [p.name for p, s in zip(files, sheets) if s.is_pending]
        partial = [f"{p.name} ({s.n_judged}/{len(s.labels)})" for p, s in zip(files, sheets) if s.is_partial]
        complete = {s.name: s.labels for s in sheets if not s.is_pending}
        res.lines.append(f"## {self.country}")
        if pending:
            res.lines.append(f"- pending (empty, ignored): {', '.join(pending)}")
        if partial:
            res.lines.append(f"- partially filled (unjudged items count as Uncertain): {', '.join(partial)}")
        if not complete:
            res.lines.append("- no completed judgments yet")
            return res
        J = pd.concat(complete, axis=1)
        kind = "annotator" if len(J.columns) == 1 else "reviewers"
        res.lines.append(f"- {len(J)} items, {len(J.columns)} completed {kind}: {', '.join(J.columns)}")
        for name in J.columns:
            c = J[name].value_counts()
            res.lines.append(f"  - {name}: Correct {c.get('Correct', 0)}, Incorrect {c.get('Incorrect', 0)}, "
                             f"Uncertain {c.get(UNCERTAIN, 0)}")
        self._inter_rater(J, res)
        self._automatic_vs_human(J, pending, res)
        self._supplementary(J, res)
        return res

    def _inter_rater(self, J: pd.DataFrame, res: ValidationResult) -> None:
        for a, b in itertools.combinations(J.columns, 2):
            po, k, n = cohen_kappa(J[a], J[b])
            conf = J[(J[a] != UNCERTAIN) & (J[b] != UNCERTAIN)]
            cpo, ck, cn = cohen_kappa(conf[a], conf[b])
            res.rows.append({"country": self.country, "comparison": f"{a} vs {b}", "n": n, "agree_3cat": 100 * po,
                             "kappa_3cat": k, "n_confident": cn, "agree_confident": 100 * cpo, "kappa_confident": ck})
            res.lines.append(f"- {a} vs {b}: 3-category agreement {100 * po:.1f}%, kappa = {k:.3f} (n={n}); "
                             f"both confident n={cn}: {100 * cpo:.1f}%, kappa = {ck:.3f}")
        if len(J.columns) == 1:
            res.lines.append("- inter-rater agreement: not available (single annotator)")

    def _automatic_vs_human(self, J: pd.DataFrame, pending: list[str], res: ValidationResult) -> None:
        sample = self.folder / "sample.csv"
        if not sample.exists():
            return
        auto = load_automatic_scores(sample)
        missing = J.index.difference(auto.index)
        if len(missing):
            res.lines.append(f"- WARNING: {len(missing)} reviewed items not found in sample.csv (id + response): "
                             f"{list(missing.get_level_values(0))[:5]}")
        auto = auto.reindex(J.index)
        ok = J.nunique(axis=1).eq(1) & J.iloc[:, 0].ne(UNCERTAIN) & auto.notna()
        human = J.iloc[:, 0][ok]
        a1, a0 = human[auto[ok] == 1], human[auto[ok] == 0]
        single = len(J.columns) == 1
        basis = "confident judgment" if single else "confident, unanimous judgment"
        res.lines += [f"- automatic score vs human (n={int(ok.sum())} items with a {basis}):",
                      f"  - automatic 'correct' confirmed by human: {(a1 == 'Correct').sum()}/{len(a1)}",
                      f"  - automatic 'incorrect' judged CORRECT by human (missed): {(a0 == 'Correct').sum()}/{len(a0)}"]
        res.rows.append({"country": self.country, "comparison": f"automatic vs {'annotator' if single else 'consensus'}",
                         "n": int(ok.sum()), "auto1_confirmed": int((a1 == "Correct").sum()), "auto1_n": len(a1),
                         "auto0_actually_correct": int((a0 == "Correct").sum()), "auto0_n": len(a0),
                         "reviewers": "+".join(J.columns), "pending": "+".join(pending)})


    def _supplementary(self, J: pd.DataFrame, res: ValidationResult) -> None:
        """Compare supplementary annotations with each reviewer and with the reviewers' consensus.
        They never enter the reported reviewer statistics above."""
        for path in self.supplementary_files():
            sheet = JudgmentSheet.load(path)
            if sheet.is_pending:
                continue
            extra = sheet.labels.reindex(J.index)
            res.lines.append(f"- supplementary (not part of the reviewer statistics): {path.name}")
            for name in J.columns:
                po, k, n = cohen_kappa(extra, J[name])
                res.lines.append(f"  - {sheet.name} vs {name}: 3-category agreement {100 * po:.1f}%, kappa = {k:.3f}")
                res.rows.append({"country": self.country, "comparison": f"supplementary: {sheet.name} vs {name}",
                                 "n": n, "agree_3cat": 100 * po, "kappa_3cat": k})
            if len(J.columns) > 1:
                cons = J.nunique(axis=1).eq(1) & J.iloc[:, 0].ne(UNCERTAIN)
                both = cons & extra.ne(UNCERTAIN)
                agree = int((extra[both] == J.iloc[:, 0][both]).sum())
                res.lines.append(f"  - agrees with the reviewers' confident consensus on {agree}/{int(both.sum())} items "
                                 f"(where {sheet.name} was also confident)")


class ValidationReport:
    def __init__(self, manual_dir: Path, countries=tuple(COUNTRIES)):
        self.manual_dir = Path(manual_dir)
        self.countries = countries

    def run(self) -> ValidationResult:
        out = ValidationResult(lines=["# Manual validation (auto-generated by `python -m blend.validation`)", ""])
        for c in self.countries:
            r = CountryValidation(c, self.manual_dir / c).run()
            out.rows += r.rows
            out.lines += r.lines + [""]
        return out

    def write(self) -> ValidationResult:
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        res = self.run()
        pd.DataFrame(res.rows).to_csv(self.manual_dir / "validation_summary.csv", index=False, float_format="%.4f")
        (self.manual_dir / "validation_summary.md").write_text("\n".join(res.lines), encoding="utf-8")
        return res


def main():
    print("\n".join(ValidationReport(Paths.from_env().manual).write().lines))


if __name__ == "__main__":
    main()
