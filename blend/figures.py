"""Poster figures, generated from results/comparison/*.csv (run `blend.analyze` first).

fig1_overall        English vs local accuracy per country x model (dumbbell, 95% Wilson CIs)
fig2_domains        per-domain gap with 95% bootstrap CIs, one panel per country
fig3_robustness     overall gap under official vs first-answer (strict) scoring
fig4_paired         share of questions correct in both / English only / local only / neither

Each figure is saved as vector PDF (for LaTeX) and 300-dpi PNG in results/figures/.

Design
  Palette        validated colours as a value object
  PosterFigures  one method per figure; data frames and output folder are injected

Usage:
    python -m blend.figures
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import COUNTRIES, DOMAIN_ORDER, DOMAIN_SHORT, Paths  # noqa: E402
from .formatting import format_p  # noqa: E402


@dataclass(frozen=True)
class Palette:
    """Validated categorical colours (CVD-safe pairs); text never uses series colours."""
    english: str = "#2a78d6"
    local: str = "#eb6834"
    models: tuple = (("GPT-4.1", "#4a3aa7"), ("Gemini 3.5 Flash-Lite", "#1baf7a"))
    official: str = "#a3a29c"
    strict: str = "#0b0b0b"
    both_correct: str = "#c8c7c1"
    both_wrong: str = "#6e6d68"
    ink: str = "#0b0b0b"
    ink2: str = "#52514e"
    grid: str = "#e4e3df"

    def model(self, name: str) -> str:
        return dict(self.models)[name]


class PosterFigures:
    DOMAINS = list(DOMAIN_ORDER)

    def __init__(self, overall: pd.DataFrame, domains: pd.DataFrame, out_dir: Path, palette: Palette = Palette(),
                 interaction: pd.DataFrame | None = None):
        self.overall = overall
        self.domains = domains
        self.out_dir = Path(out_dir)
        self.c = palette
        self.interaction = interaction  # optional: GEE language x domain tests, printed above each domain panel

    # ---------------------------------------------------------------- helpers
    def apply_style(self) -> None:
        c = self.c
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 24, "axes.titlesize": 26, "axes.titleweight": "bold",
            "axes.labelsize": 24, "xtick.labelsize": 22, "ytick.labelsize": 24, "legend.fontsize": 23,
            "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
            "axes.edgecolor": c.ink2, "axes.labelcolor": c.ink, "xtick.color": c.ink2, "ytick.color": c.ink,
            "text.color": c.ink, "axes.grid": True, "axes.grid.axis": "x", "grid.color": c.grid, "grid.linewidth": 1,
            "ytick.major.size": 0, "savefig.facecolor": "white", "figure.facecolor": "white",
        })

    def save(self, fig, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pdf = self.out_dir / f"{name}.pdf"
        fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})  # byte-stable
        fig.savefig(self.out_dir / f"{name}.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)
        print("wrote", pdf)
        return pdf

    @staticmethod
    def row_label(r) -> str:
        return f"{r.country} · {r.model}"

    def _official(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.scoring == "official"].reset_index(drop=True)

    def _emphasis(self, p: float) -> dict:
        return {"color": self.c.ink if p < .05 else self.c.ink2, "fontweight": "bold" if p < .05 else "normal"}

    # ---------------------------------------------------------------- fig 1
    def overall_accuracy(self) -> Path:
        """English vs local accuracy per setting x model (dumbbell, 95% Wilson CIs)."""
        c, d = self.c, self._official(self.overall)
        fig, ax = plt.subplots(figsize=(12, 7.2))
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows()):
            # English slightly above, local slightly below, so the two confidence intervals never overlap
            for acc, lo, hi, col, dy in [(r.acc_en, r.acc_en_lo, r.acc_en_hi, c.english, 0.17),
                                         (r.acc_loc, r.acc_loc_lo, r.acc_loc_hi, c.local, -0.17)]:
                ax.errorbar(acc, yi + dy, xerr=[[acc - lo], [hi - acc]], fmt="o", ms=15, color=col, mec="white",
                            mew=2, elinewidth=3, capsize=0, zorder=3)
                ax.text(hi + 0.4, yi + dy, f"{acc:.1f}", va="center", ha="left", fontsize=21, color=c.ink)
            ax.text(84, yi, f"{r.gap_pp:+.1f} pp\n{format_p(r.mcnemar_p)}", va="center", ha="left", fontsize=22,
                    **self._emphasis(r.mcnemar_p))
        ax.set_yticks(y, [self.row_label(r) for _, r in d.iterrows()])
        ax.set_xlim(45, 83)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_xlabel("Accuracy (% correct, 95% CI); bold gap = significant (p < .05)")
        ax.plot([], [], "o", color=c.english, ms=12, label="English prompt")
        ax.plot([], [], "o", color=c.local, ms=12, label="Local-language prompt")
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False)
        ax.text(84, len(d) - 0.45, "EN − local", fontsize=22, color=c.ink2, ha="left", va="bottom")
        return self.save(fig, "fig1_overall")

    # ---------------------------------------------------------------- fig 2
    def domain_gaps(self) -> Path:
        """Per-domain gap with 95% bootstrap CIs, one panel per setting; filled = Holm-significant."""
        c, d = self.c, self.domains[self.domains.scoring == "official"]
        models = list(dict.fromkeys(d.model))
        fig, axes = plt.subplots(1, len(COUNTRIES), figsize=(13, 8.4), sharey=True, sharex=True)
        y = np.arange(len(self.DOMAINS))
        off = dict(zip(models, np.linspace(-0.17, 0.17, len(models))))
        for ax, (country, spec) in zip(np.atleast_1d(axes), COUNTRIES.items()):
            ax.axvline(0, color=c.ink2, lw=1.2, zorder=1)
            for m in models:
                s = d[(d.country == country) & (d.model == m)].set_index("domain").reindex(self.DOMAINS)
                sig = s.mcnemar_p_holm < .05
                ax.errorbar(s.gap_pp, y + off[m], xerr=[s.gap_pp - s.gap_lo, s.gap_hi - s.gap_pp], fmt="none",
                            ecolor=c.model(m), elinewidth=2, zorder=2)
                ax.scatter(s.gap_pp[~sig], (y + off[m])[~sig.to_numpy()], s=110, facecolor="white",
                           edgecolor=c.model(m), linewidth=2.2, zorder=3)
                ax.scatter(s.gap_pp[sig], (y + off[m])[sig.to_numpy()], s=150, color=c.model(m),
                           edgecolor=c.model(m), linewidth=2.2, zorder=3)
            ax.set_title(country, loc="left", pad=62 if self.interaction is not None else 6)
            if self.interaction is not None:  # RQ2 test result, so it can be read from the figure itself
                g = self.interaction[(self.interaction.country == country) & (self.interaction.scoring == "official")]
                parts = [f"{m.split()[0]} {format_p(g[g.model == m].gee_p.iloc[0]).replace(' ', '')}" for m in models]
                ax.text(0, 1.015, "Language × domain (GEE):\n" + "  ·  ".join(parts), transform=ax.transAxes,
                        fontsize=19, color=c.ink2, va="bottom", ha="left")
        axes[0].set_yticks(y, [DOMAIN_SHORT.get(x, x) for x in self.DOMAINS])
        axes[0].invert_yaxis()
        for m in models:
            axes[0].scatter([], [], s=110, color=c.model(m), label=m)
        axes[0].scatter([], [], s=110, facecolor="white", edgecolor=c.ink2, linewidth=2, label="open = not significant (Holm)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                   frameon=False)
        fig.supxlabel("← local better     Gap: English − local (pp, 95% CI)     English better →", fontsize=24)
        fig.tight_layout()
        return self.save(fig, "fig2_domains")

    # ---------------------------------------------------------------- fig 3
    def sensitivity(self) -> Path:
        """Overall gap under official (primary) vs first-answer (sensitivity) scoring."""
        c = self.c
        o = self._official(self.overall)
        s = self.overall[self.overall.scoring == "strict"].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 5.2))
        y = np.arange(len(o))[::-1]
        ax.axvline(0, color=c.ink2, lw=1.2)
        for yi, (_, a), (_, b) in zip(y, o.iterrows(), s.iterrows()):
            for r, col, dy in [(a, c.official, 0.16), (b, c.strict, -0.16)]:
                ax.errorbar(r.gap_pp, yi + dy, xerr=[[r.gap_pp - r.gap_lo], [r.gap_hi - r.gap_pp]], fmt="o", ms=12,
                            color=col, mec="white", mew=1.5, elinewidth=2, zorder=3)
            ax.text(9.5, yi, f"{b.gap_pp:+.1f} pp, {format_p(b.mcnemar_p)}", va="center", fontsize=17,
                    **self._emphasis(b.mcnemar_p))
        ax.set_yticks(y, [self.row_label(r) for _, r in o.iterrows()])
        ax.set_xlim(-16, 9)
        ax.set_ylim(-0.7, len(o) - 0.3)
        ax.set_xlabel("Gap: English − local (pp, 95% CI)     ← local better")
        ax.plot([], [], "o", color=c.official, ms=11, label="official BLEnD scoring (primary)")
        ax.plot([], [], "o", color=c.strict, ms=11, label="first answer only (sensitivity)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        ax.text(9.5, len(o) - 0.35, "strict", fontsize=17, color=c.ink2, va="bottom")
        return self.save(fig, "fig3_robustness")

    # ---------------------------------------------------------------- fig 4
    def paired_outcomes(self) -> Path:
        """Share of questions correct in both / English only / local only / neither."""
        c, d = self.c, self._official(self.overall)
        parts = [("both_correct", "Both correct", c.both_correct), ("en_only", "English only", c.english),
                 ("loc_only", "Local only", c.local), ("both_wrong", "Both wrong", c.both_wrong)]
        fig, ax = plt.subplots(figsize=(12, 5.4))
        y = np.arange(len(d))[::-1]
        left = np.zeros(len(d))
        for col, lab, colour in parts:
            v = (100 * d[col] / d.n).to_numpy(float)
            ax.barh(y, v, left=left, color=colour, height=0.62, label=lab, edgecolor="white", linewidth=2)
            for yi, l, vv in zip(y, left, v):
                if vv >= 6:
                    ax.text(l + vv / 2, yi, f"{vv:.0f}%", ha="center", va="center", fontsize=22,
                            color="white" if colour != c.both_correct else c.ink, fontweight="bold")
            left += v
        ax.set_yticks(y, [self.row_label(r) for _, r in d.iterrows()])
        ax.set_xlim(0, 100)
        ax.grid(False)
        ax.set_xlabel("Share of paired questions (%)")
        ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0))
        return self.save(fig, "fig4_paired")

    def draw_all(self) -> list[Path]:
        self.apply_style()
        return [self.overall_accuracy(), self.domain_gaps(), self.sensitivity(), self.paired_outcomes()]


def main():
    paths = Paths.from_env()
    PosterFigures(pd.read_csv(paths.comparison / "overall.csv"), pd.read_csv(paths.comparison / "domains.csv"),
                  paths.figures, interaction=pd.read_csv(paths.comparison / "interaction.csv")).draw_all()


if __name__ == "__main__":
    main()
