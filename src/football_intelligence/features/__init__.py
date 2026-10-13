"""Feature-engineering subpackage for pre-match predictions.

This package provides leakage-safe feature extraction from the project's
SQLite database.  Features are computed using only match results that were
known **before** the target fixture's scheduled date, preventing both
target-leakage and future-information leakage.

Public API
----------
:class:`FeatureEngineer`
    The main engine.  Instantiate with a set of lookback windows and call
    :meth:`compute_features` (single match) or :meth:`compute_all_features`
    (every match in the database).
"""
from __future__ import annotations

from .engineering import FeatureEngineer, FeatureSpec

__all__ = ["FeatureEngineer", "FeatureSpec"]
