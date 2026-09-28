"""Shared presentation helpers (used by the summary and the figures)."""


def format_p(p: float) -> str:
    """APA-style p-value: 'p < .001' or 'p = .015'."""
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")
