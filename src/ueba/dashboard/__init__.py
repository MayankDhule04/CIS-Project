"""Dashboard package for the UEBA Insider Threat Detection system.

The Streamlit entry point lives in the project-root ``app.py``; this package
holds the *pure*, Streamlit-free helpers that ``app.py`` imports — ranking users
by Weekly_Risk, filtering user-days by a risk threshold, and reading the
pre-computed ``risk.parquet`` artifact. Keeping this logic here (with no
``import streamlit``) means the Phase 6 property tests can exercise the ranking
and threshold rules headlessly, without spinning up a Streamlit runtime.
"""

from .logic import (
    RISK_PARQUET_MISSING_MESSAGE,
    alerts_per_day,
    available_day_bounds,
    filter_by_threshold,
    load_risk,
    rank_users,
    risk_parquet_path,
    user_daily_timeline,
    user_peak_day,
    user_reasons,
)

__all__ = [
    "RISK_PARQUET_MISSING_MESSAGE",
    "alerts_per_day",
    "available_day_bounds",
    "filter_by_threshold",
    "load_risk",
    "rank_users",
    "risk_parquet_path",
    "user_daily_timeline",
    "user_peak_day",
    "user_reasons",
]
