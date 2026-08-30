"""Shared multiple-comparison correction for hypothesis tests (H1, H2 -- karar-notu SS6)."""

from __future__ import annotations


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        candidate = (m - rank) * pvalues[i]
        running_max = max(running_max, candidate)
        adjusted[i] = min(running_max, 1.0)
    return adjusted
