"""
Project-specific exception hierarchy.

All custom errors inherit from :class:`FootballPredictError` so consumers
can catch the whole family with one ``except`` clause. Exceptions land
incrementally as T2.1 commits arrive — see git log for which commit owns
each subclass.
"""

from __future__ import annotations


class FootballPredictError(Exception):
    """Base class for all project-specific errors."""


class QualityGateFailure(FootballPredictError):
    """One or more quality gates rejected the trained model.

    Carries the full :class:`evaluation.cv_report.GatesSection` so callers
    can render a human-readable breakdown without re-loading the report.
    The exception message is intentionally short; use
    :meth:`verbose_breakdown` for the full listing.
    """

    def __init__(self, gates) -> None:  # GatesSection — not typed to avoid cycle
        self.gates = gates
        super().__init__(self._summary())

    def _summary(self) -> str:
        n = len(self.gates.failures)
        return f"Quality gates failed: {n} failure{'s' if n != 1 else ''}"

    def verbose_breakdown(self) -> str:
        g = self.gates
        lines = ["Quality gate breakdown:"]
        for failure in g.failures:
            lines.append(f"  - {failure}")
        lines.append("")
        lines.append(
            f"Thresholds: max_rps={g.max_rps}, max_brier={g.max_brier}, "
            f"min_draw_f1={g.min_draw_f1}"
        )
        lines.append(
            f"CV means:   rps={g.cv_mean_rps:.4f}, "
            f"brier={g.cv_mean_brier:.4f}, draw_f1={g.cv_mean_draw_f1:.4f}"
        )
        lines.append(
            f"Holdout:    rps={g.holdout_rps:.4f}, "
            f"brier={g.holdout_brier:.4f}, draw_f1={g.holdout_draw_f1:.4f}"
        )
        return "\n".join(lines)
