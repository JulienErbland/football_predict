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
