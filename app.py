"""Streamlit dashboard for the UEBA Insider Threat Detection system.

Two tabs:

* **Overview** - a plain-language landing page explaining what the system does,
  how it works in five steps, the three scoring signals and how they fuse, what
  each dashboard panel means, and an honest note on scope.
* **Live dashboard** - the analyst view: summary metrics, a "most risky users"
  table (with each user's alert-day count and strongest reason), the most common
  anomaly triggers, a per-user daily-risk timeline with its "why flagged"
  features, and an alerts-per-day chart.

Data source. By default the dashboard reads the pre-computed ``risk.parquet``
produced by the pipeline (``python scripts/run_eval.py``). You can also drop
CERT-schema CSVs into the sidebar uploader (``logon.csv``, ``device.csv``,
``http.csv``, ``email.csv``, optionally ``file.csv``/``roles.csv``) and the full
feature -> baseline -> scoring pipeline runs on them in-app. Uploads suit a
sample or synthetic export; the full ~20 GB dataset should live in ``data/raw``
and be processed with the scripts.

The dashboard performs no scoring at render time unless you upload files, makes
no network calls, and never displays ground-truth labels (those are
evaluation-only). Run with::

    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ueba.config import Config, load_config
from src.ueba.dashboard.logic import (
    RISK_PARQUET_MISSING_MESSAGE,
    alerts_per_day,
    available_day_bounds,
    filter_by_threshold,
    load_risk,
    rank_users,
    user_daily_timeline,
    user_peak_day,
    user_reasons,
)

# How many users to list in the "most risky users" table.
_TOP_USERS = 15

# A small palette reused for the data-derived trigger bars.
_BAR_COLORS = ["#f43f5e", "#f59e0b", "#38bdf8", "#22c55e", "#a855f7"]

# Raw CERT files the pipeline knows how to read when scoring an upload.
_KNOWN_UPLOAD_FILES = {
    "logon.csv", "device.csv", "http.csv", "email.csv",
    "file.csv", "roles.csv", "labels.csv",
}


# -----------------------------------------------------------------------------
# Styling (cosmetic only - a dark analyst theme)
# -----------------------------------------------------------------------------
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
:root{
  --bg:#0b0f17; --surface:#121a28; --border:rgba(148,163,184,0.15);
  --accent:#38bdf8; --good:#22c55e; --warn:#f59e0b; --bad:#f43f5e;
  --text:#e2e8f0; --muted:#94a3b8; --dim:#64748b;
  --mono:'JetBrains Mono',monospace; --sans:'Plus Jakarta Sans',sans-serif;
}
.stApp{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
.main .block-container{max-width:1400px!important;padding-top:1.4rem!important;}
#MainMenu,.stDeployButton,footer{display:none!important;}
.app-title{font-size:1.6rem;font-weight:800;color:#fff;margin:0;}
.app-sub{color:var(--muted);font-size:0.9rem;margin-top:0.3rem;max-width:860px;}
.status-line{font-family:var(--mono);font-size:0.72rem;color:var(--dim);margin-top:0.5rem;}
.status-line b{color:var(--accent);font-weight:600;}
.source-pill{font-family:var(--mono);font-size:0.72rem;color:var(--muted);margin:0.6rem 0 0.2rem 0;}
.source-pill b{color:var(--good);}
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.1rem 0;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.1rem 1.2rem;}
.metric-label{font-family:var(--mono);font-size:0.66rem;font-weight:700;letter-spacing:0.1em;color:var(--dim);text-transform:uppercase;margin-bottom:0.5rem;}
.metric-value{font-family:var(--mono);font-size:1.8rem;font-weight:800;color:#fff;line-height:1;}
.metric-unit{font-size:0.8rem;color:var(--muted);font-weight:500;margin-left:0.35rem;}
.metric-sub{font-size:0.74rem;color:var(--dim);margin-top:0.45rem;}
.section-title{font-size:1.1rem;font-weight:700;color:#fff;margin:1.5rem 0 0.2rem 0;}
.section-sub{font-size:0.76rem;color:var(--dim);margin-bottom:0.8rem;}
.track{width:100%;height:5px;background:#0b1120;border-radius:3px;overflow:hidden;}
.bar{height:100%;border-radius:3px;}
.driver-row{margin-bottom:0.75rem;}
.driver-meta{display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.3rem;}
.driver-name{color:#cbd5e1;} .driver-pct{font-family:var(--mono);font-weight:700;color:#fff;}
.sidebar-title{font-family:var(--mono);font-size:0.66rem;font-weight:700;letter-spacing:0.1em;color:var(--dim);text-transform:uppercase;margin:0.75rem 0 0.5rem 0;}
.about-box{font-size:0.78rem;color:var(--muted);line-height:1.55;}
.about-box b{color:#cbd5e1;}
div[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden!important;}
.hero{background:linear-gradient(135deg,rgba(56,189,248,0.08),rgba(34,197,94,0.05));border:1px solid var(--border);border-radius:14px;padding:1.9rem 2.1rem;margin:0.5rem 0 1.25rem 0;}
.hero-eyebrow{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.18em;color:var(--accent);font-weight:700;}
.hero-title{font-size:1.75rem;font-weight:800;color:#fff;margin:0.5rem 0;letter-spacing:-0.02em;}
.hero-text{color:var(--muted);font-size:0.95rem;line-height:1.65;max-width:840px;}
.step-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:0.85rem;margin-bottom:0.5rem;}
.step-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem;}
.step-num{width:26px;height:26px;border-radius:7px;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.3);color:var(--accent);font-family:var(--mono);font-weight:700;display:flex;align-items:center;justify-content:center;font-size:0.8rem;margin-bottom:0.6rem;}
.step-name{font-weight:700;color:#fff;font-size:0.92rem;margin-bottom:0.3rem;}
.step-desc{color:var(--muted);font-size:0.78rem;line-height:1.5;}
.signal-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
.signal-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.1rem 1.2rem;}
.signal-tag{font-family:var(--mono);font-size:0.62rem;font-weight:700;letter-spacing:0.08em;padding:2px 7px;border-radius:4px;text-transform:uppercase;}
.signal-name{font-weight:700;color:#fff;font-size:1rem;margin:0.55rem 0 0.35rem 0;}
.signal-desc{color:var(--muted);font-size:0.8rem;line-height:1.5;}
.signal-weight{font-family:var(--mono);color:var(--accent);font-weight:700;font-size:0.78rem;margin-top:0.55rem;}
.formula-box{background:#0b1120;border:1px solid var(--border);border-radius:10px;padding:1.05rem 1.3rem;margin:1rem 0;font-family:var(--mono);color:#e2e8f0;font-size:0.92rem;text-align:center;line-height:1.7;}
.formula-box b{color:var(--good);}
.formula-box .muted{color:var(--dim);font-size:0.8rem;}
.panel-explain{display:grid;grid-template-columns:repeat(2,1fr);gap:0.75rem;}
.pe-item{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.8rem 1rem;}
.pe-title{font-weight:700;color:#fff;font-size:0.85rem;margin-bottom:0.25rem;}
.pe-desc{color:var(--muted);font-size:0.78rem;line-height:1.45;}
.note-box{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.25);border-radius:10px;padding:1rem 1.2rem;color:var(--muted);font-size:0.82rem;line-height:1.6;margin-top:0.5rem;}
.note-box b{color:#fbbf24;}
</style>
"""


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading risk data…")
def _load_risk_cached(processed_dir: str) -> pd.DataFrame:
    """Load ``risk.parquet`` once and cache it across Streamlit re-runs."""
    config = load_config()
    return load_risk(config)


def _load_config_safe() -> Config:
    """Load the typed config, or stop with a clear message if it is invalid."""
    try:
        return load_config()
    except Exception as exc:
        st.error(f"Could not load configuration: {exc}")
        st.stop()


def _build_risk_from_uploads(files, config: Config) -> pd.DataFrame:
    """Run the real pipeline on uploaded CERT CSVs and return the risk frame.

    The uploaded files are written (by their original names) into a temporary
    raw directory; a copy of the config is pointed at that directory and a
    temporary processed directory; then the Feature_Extractor and Scoring_Engine
    run exactly as they do on disk data. Returns an empty frame when the uploads
    yield no scorable user-days.
    """
    # Imported locally so merely viewing pre-computed data never pulls in the
    # model stack.
    from src.ueba.features import extract_features
    from src.ueba.scoring import compute_risk

    tmp_root = tempfile.mkdtemp(prefix="ueba_upload_")
    raw_dir = os.path.join(tmp_root, "raw")
    processed_dir = os.path.join(tmp_root, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    for uploaded in files:
        name = os.path.basename(uploaded.name)
        with open(os.path.join(raw_dir, name), "wb") as handle:
            handle.write(uploaded.getbuffer())

    cfg = dataclasses.replace(
        config, raw_data_dir=raw_dir, processed_dir=processed_dir
    )
    features = extract_features(cfg)
    if features is None or features.empty:
        return pd.DataFrame()
    return compute_risk(features, cfg)


def _upload_signature(files) -> tuple:
    """A cheap, hashable signature of an upload so the pipeline runs only once."""
    return tuple(sorted((f.name, f.size) for f in files))


def _risk_from_uploads(files, config: Config) -> pd.DataFrame | None:
    """Score uploaded files, caching the result in session state by signature."""
    signature = _upload_signature(files)
    if st.session_state.get("_upload_sig") == signature:
        return st.session_state.get("_upload_risk")

    try:
        with st.spinner("Running the pipeline on your uploaded logs…"):
            risk = _build_risk_from_uploads(files, config)
    except Exception as exc:  # surface pipeline errors instead of crashing
        st.session_state["_upload_sig"] = signature
        st.session_state["_upload_risk"] = None
        st.sidebar.error(f"Could not score the upload: {exc}")
        return None

    st.session_state["_upload_sig"] = signature
    st.session_state["_upload_risk"] = risk
    return risk


def resolve_data_source(config: Config) -> tuple[pd.DataFrame | None, str]:
    """Render the sidebar data-source picker and return ``(risk_df, label)``.

    If the analyst drops CERT CSVs they are scored by the real pipeline and used;
    otherwise the pre-computed ``risk.parquet`` is loaded. Returns ``(None, "")``
    when neither source is available so the caller can show guidance.
    """
    st.sidebar.markdown('<div class="sidebar-title">Data source</div>', unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader(
        "Analyze your own CSVs",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Drop CERT-schema logs: logon.csv, device.csv, http.csv, email.csv "
            "(and optionally file.csv, roles.csv). The full feature -> baseline "
            "-> scoring pipeline runs on them. Best for a sample or synthetic "
            "export; the full ~20 GB dataset belongs in data/raw via the scripts."
        ),
    )

    if uploaded:
        recognized = [f for f in uploaded if os.path.basename(f.name) in _KNOWN_UPLOAD_FILES]
        if not recognized:
            st.sidebar.warning(
                "None of the uploaded files match the expected CERT names "
                "(logon.csv, device.csv, http.csv, email.csv). Showing built-in data."
            )
        else:
            risk = _risk_from_uploads(recognized, config)
            if risk is not None and not risk.empty:
                names = ", ".join(sorted(os.path.basename(f.name) for f in recognized))
                st.sidebar.success(
                    f"Scored {risk['user'].nunique()} users from your upload "
                    f"({names})."
                )
                return risk, "your uploaded logs"
            if risk is not None:  # ran but produced nothing scorable
                st.sidebar.warning(
                    "The upload produced no scorable user-days (too little "
                    "history for a baseline?). Showing built-in data."
                )

    try:
        return _load_risk_cached(config.processed_dir), "data/processed/risk.parquet"
    except FileNotFoundError:
        return None, ""


def _apply_date_range(
    risk_df: pd.DataFrame, date_range: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    """Return the user-days whose ``day`` falls within ``date_range`` (inclusive)."""
    if "day" not in risk_df.columns or risk_df.empty:
        return risk_df
    start, end = date_range
    days = pd.to_datetime(risk_df["day"]).dt.normalize()
    mask = (days >= pd.Timestamp(start).normalize()) & (
        days <= pd.Timestamp(end).normalize()
    )
    return risk_df.loc[mask]


def _feature_driver_counts(df: pd.DataFrame) -> Counter:
    """Count how often each feature appears in the stored ``reasons`` records."""
    counter: Counter = Counter()
    if "reasons" not in df.columns:
        return counter
    for cell in df["reasons"]:
        if cell is None or isinstance(cell, float):
            continue
        try:
            items = list(cell)
        except TypeError:
            continue
        for item in items:
            if item is None:
                continue
            try:
                feature = item["feature"]
            except (KeyError, TypeError, IndexError):
                continue
            if feature:
                counter[str(feature)] += 1
    return counter


def _top_reason(risk_df: pd.DataFrame, user: str) -> str:
    """Return the single strongest feature on a user's peak-risk day, or '-'."""
    peak = user_peak_day(risk_df, user)
    if peak is None:
        return "-"
    reasons = user_reasons(risk_df, user, peak)
    return str(reasons[0]["feature"]) if reasons else "-"


# -----------------------------------------------------------------------------
# Header + Overview (landing)
# -----------------------------------------------------------------------------
def render_header(config: Config) -> None:
    """Render an honest title, one-line description, and true status facts."""
    w = config.weights
    b = config.baseline
    st.markdown(
        f"""
        <h1 class="app-title">UEBA Insider Threat Detection</h1>
        <p class="app-sub">
          Users ranked by how far their daily behaviour departs from their own
          and their peers' baseline - with a plain-language overview and a
          read-only analyst dashboard over the risk scores.
        </p>
        <div class="status-line">
          DATASET <b>CMU/SEI CERT r4.2</b> &nbsp;&middot;&nbsp;
          OFFLINE (no network calls) &nbsp;&middot;&nbsp;
          FUSION WEIGHTS <b>self {w.self_w} / iso {w.iso_w} / peer {w.peer_w}</b>
          &nbsp;&middot;&nbsp; BASELINE <b>{b.min_days}-{b.max_days} days</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(config: Config) -> None:
    """Render the plain-language landing page explaining how the system works."""
    w = config.weights
    b = config.baseline

    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">USER &amp; ENTITY BEHAVIOUR ANALYTICS</div>
          <div class="hero-title">Catch the insider by learning everyone's normal</div>
          <p class="hero-text">
            Insider threats don't match fixed rules - a trusted employee slowly
            starts behaving unlike themselves. So instead of rules, this system
            <b>learns each person's normal day-to-day behaviour</b> from their
            activity logs, then flags the days that drift furthest from that
            baseline, ranks the riskiest people for an analyst, and shows exactly
            which behaviours triggered each flag.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">How it works</div>
        <div class="section-sub">Raw logs in, a ranked and explained risk list out - in five steps.</div>
        """,
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Ingest", "Read the raw CERT logs (logon, device, web, email, file) in chunks and compress them to one row per user per day."),
        ("2", "Features", "Turn each user-day into 15 behavioural numbers: after-hours logons, USB use, job-site visits, uploads, external emails, and more."),
        ("3", "Baseline", f"Learn each person's - and their team's - normal from their first {b.min_days}-{b.max_days} days. Only this early history trains the baseline."),
        ("4", "Score", "Compare every later day to that baseline three different ways and fuse them into one 0-100 daily risk score."),
        ("5", "Rank &amp; explain", "Sum each user's last 7 daily scores, rank everyone riskiest-first, and store the top-3 features behind every flag."),
    ]
    cards = "".join(
        f"""<div class="step-card"><div class="step-num">{n}</div>
        <div class="step-name">{name}</div><div class="step-desc">{desc}</div></div>"""
        for n, name, desc in steps
    )
    st.markdown(f'<div class="step-grid">{cards}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">The three signals</div>
        <div class="section-sub">Each day is judged three ways, because no single signal catches every kind of insider.</div>
        """,
        unsafe_allow_html=True,
    )
    signals = [
        ("SELF", "#38bdf8", "rgba(56,189,248,0.15)", "Self z-score",
         "How unusual today is versus this person's OWN history. Someone who never touches a USB drive suddenly copying 30 files stands out sharply.",
         f"weight {w.self_w}"),
        ("MODEL", "#f59e0b", "rgba(245,158,11,0.15)", "Isolation Forest",
         "An unsupervised machine-learning model that flags days which don't fit the normal pattern across all 15 features at once.",
         f"weight {w.iso_w}"),
        ("PEER", "#22c55e", "rgba(34,197,94,0.15)", "Peer z-score",
         "How unusual today is versus others in the SAME job role. Catches people who are quietly odd for their team, not just for themselves.",
         f"weight {w.peer_w}"),
    ]
    sig_cards = "".join(
        f"""<div class="signal-card">
          <span class="signal-tag" style="background:{bg};color:{color};">{tag}</span>
          <div class="signal-name">{name}</div>
          <div class="signal-desc">{desc}</div>
          <div class="signal-weight">{weight}</div>
        </div>"""
        for tag, color, bg, name, desc, weight in signals
    )
    st.markdown(f'<div class="signal-grid">{sig_cards}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="formula-box">
          Daily risk = 100 &times; ( <b>{w.self_w}</b>&middot;self + <b>{w.iso_w}</b>&middot;iso + <b>{w.peer_w}</b>&middot;peer )
          <div class="muted">example: self 0.8, iso 0.6, peer 0.4 &rarr; 66 / 100 &nbsp;&middot;&nbsp;
          weekly risk = sum of the last 7 daily scores, so a slow build-up outranks one noisy day</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">What you see in the dashboard</div>
        <div class="section-sub">Open the Live dashboard tab - every panel is computed from the real risk scores.</div>
        """,
        unsafe_allow_html=True,
    )
    panels = [
        ("Summary metrics", "Users monitored, how many user-days clear the alert threshold, how many distinct users are flagged, and the single highest weekly risk."),
        ("Most risky users", "The triage table - who to look at first, each with their alert-day count and the single strongest reason."),
        ("Top triggers", "The behaviours behind the most alerts right now, tallied from the stored reasons - not a fixed list."),
        ("Per-user drill-down", "One person's risk over time, plus the exact features (value vs baseline, z-score) behind a chosen flagged day."),
        ("Alerts per day", "When alert volume spikes over the timeline, so you can spot a campaign or a quiet build-up."),
        ("Bring your own data", "Drop CERT-schema CSVs in the sidebar to score your own logs through the same pipeline, or use the built-in results."),
        ("Near-real-time &amp; replay", "New days can be scored as they arrive (daily micro-batch via scripts/live_watch.py); Replay mode steps through recorded days to show risk building up."),
    ]
    pe = "".join(
        f'<div class="pe-item"><div class="pe-title">{t}</div><div class="pe-desc">{d}</div></div>'
        for t, d in panels
    )
    st.markdown(f'<div class="panel-explain">{pe}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note-box">
          <b>What this is - and isn't.</b> It's a decision-support tool that
          ranks anomalies for a human analyst to review, not an automatic verdict
          on anyone. It runs offline over the scores, with no live feed.
          Ground-truth insider labels are used only to measure accuracy during
          evaluation - never to train the model, and never shown on this screen.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Dashboard panels (simplified)
# -----------------------------------------------------------------------------
def render_metric_cards(
    total_users: int,
    total_days: int,
    n_alert_days: int,
    n_flagged_users: int,
    peak_weekly: float,
    peak_user: str,
    threshold: float,
) -> None:
    """Render four summary cards, every value computed from the data in view."""
    flagged_pct = (n_flagged_users / total_users * 100.0) if total_users else 0.0
    alert_pct = (n_alert_days / total_days * 100.0) if total_days else 0.0
    st.markdown(
        f"""
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Monitored users</div>
            <div><span class="metric-value">{total_users:,}</span><span class="metric-unit">users</span></div>
            <div class="metric-sub">{total_days:,} user-days in view</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Alert user-days</div>
            <div><span class="metric-value">{n_alert_days:,}</span><span class="metric-unit">&ge; {threshold:.1f}</span></div>
            <div class="metric-sub">{alert_pct:.1f}% of user-days in view</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Flagged users</div>
            <div><span class="metric-value">{n_flagged_users:,}</span><span class="metric-unit">users</span></div>
            <div class="metric-sub">{flagged_pct:.1f}% have at least one alert day</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Highest weekly risk</div>
            <div><span class="metric-value">{peak_weekly:.0f}</span></div>
            <div class="metric-sub">{peak_user}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_users(risk_df: pd.DataFrame, threshold: float) -> None:
    """Render one clean table of the riskiest users with alert days and top reason."""
    st.markdown(
        """
        <div class="section-title">Most risky users</div>
        <div class="section-sub">Ranked by peak weekly risk. "Top reason" is the strongest feature on the user's worst day.</div>
        """,
        unsafe_allow_html=True,
    )
    filtered = filter_by_threshold(risk_df, threshold)
    if filtered.empty:
        st.info("No user-days meet the current date range and threshold. Lower the threshold or widen the dates.")
        return

    ranking = rank_users(filtered).head(_TOP_USERS)
    alert_counts = filtered.groupby(filtered["user"].astype("string")).size()

    rows = []
    for position, (_, row) in enumerate(ranking.iterrows(), start=1):
        user = str(row["user"])
        rows.append(
            {
                "#": position,
                "User": user,
                "Weekly risk": round(float(row["weekly_risk"]), 1),
                "Alert days": int(alert_counts.get(user, 0)),
                "Top reason": _top_reason(risk_df, user),
            }
        )
    table = pd.DataFrame(rows)
    max_risk = float(table["Weekly risk"].max()) if not table.empty else 1.0

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Weekly risk": st.column_config.ProgressColumn(
                "Weekly risk",
                help="Peak trailing-7-day sum of daily risk.",
                format="%.1f",
                min_value=0.0,
                max_value=max(1.0, max_risk),
            ),
            "Alert days": st.column_config.NumberColumn("Alert days", format="%d"),
            "Top reason": st.column_config.TextColumn("Top reason", width="medium"),
        },
    )


def render_top_triggers(alert_df: pd.DataFrame) -> None:
    """Render the five most common 'why flagged' features as simple bars."""
    st.markdown(
        """
        <div class="section-title">Top triggers</div>
        <div class="section-sub">Most common reasons behind current alerts.</div>
        """,
        unsafe_allow_html=True,
    )
    counts = _feature_driver_counts(alert_df)
    if not counts:
        st.info("No reasons among current alerts.")
        return
    total = sum(counts.values())
    rows = []
    for i, (feature, count) in enumerate(counts.most_common(5)):
        pct = count / total * 100.0
        color = _BAR_COLORS[i % len(_BAR_COLORS)]
        rows.append(
            f"""
            <div class="driver-row">
              <div class="driver-meta"><span class="driver-name">{feature}</span>
              <span class="driver-pct">{pct:.0f}%</span></div>
              <div class="track"><div class="bar" style="width:{pct:.0f}%;background:{color};"></div></div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_drilldown(risk_df: pd.DataFrame, threshold: float) -> None:
    """Render the per-user drill-down: daily-risk timeline and why-flagged table."""
    ranking = rank_users(risk_df)
    st.markdown(
        """
        <div class="section-title">Per-user drill-down</div>
        <div class="section-sub">Pick a user to see their daily risk over time and why a given day was flagged.</div>
        """,
        unsafe_allow_html=True,
    )
    if ranking.empty:
        st.info("No users in the current date range.")
        return

    users = ranking["user"].tolist()
    col_user, col_day = st.columns([1, 1])
    with col_user:
        selected_user = st.selectbox("User", options=users, index=0)

    timeline = user_daily_timeline(risk_df, selected_user)
    if timeline.empty:
        st.info("The selected user has no user-days in the current date range.")
        return

    peak_day = user_peak_day(risk_df, selected_user)
    day_options = timeline["day"].tolist()
    default_index = day_options.index(peak_day) if peak_day in day_options else 0
    with col_day:
        selected_day = st.selectbox(
            "Day to explain",
            options=day_options,
            index=default_index,
            format_func=lambda d: pd.Timestamp(d).date().isoformat(),
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["day"],
            y=timeline["daily_risk"],
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.12)",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        )
    )
    if threshold > 0:
        fig.add_hline(y=threshold, line_dash="dot", line_color="#f43f5e")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=10, b=20),
        height=240,
        showlegend=False,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Daily risk", gridcolor="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True)

    reasons = user_reasons(risk_df, selected_user, selected_day)
    date_str = pd.Timestamp(selected_day).date().isoformat()
    st.caption(f"Why flagged - {selected_user} on {date_str}")
    if not reasons:
        st.info("No contributing reasons were stored for this user-day.")
        return
    reasons_table = pd.DataFrame(reasons).rename(
        columns={
            "feature": "Feature",
            "value": "Value",
            "baseline_mean": "Baseline mean",
            "z_score": "Z-score",
        }
    )[["Feature", "Value", "Baseline mean", "Z-score"]]
    st.dataframe(
        reasons_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Value": st.column_config.NumberColumn("Value", format="%.2f"),
            "Baseline mean": st.column_config.NumberColumn("Baseline mean", format="%.2f"),
            "Z-score": st.column_config.NumberColumn("Z-score", format="%+.2f"),
        },
    )


def render_alerts_per_day(risk_df: pd.DataFrame, threshold: float) -> None:
    """Render a simple alerts-per-day chart (user-days at/above the threshold)."""
    st.markdown(
        """
        <div class="section-title">Alerts per day</div>
        <div class="section-sub">How many user-days clear the risk threshold over time.</div>
        """,
        unsafe_allow_html=True,
    )
    per_day = alerts_per_day(risk_df, threshold)
    if per_day.empty:
        st.info("No alerts at the current threshold. Lower it to see activity.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=per_day["day"],
            y=per_day["alert_count"],
            marker=dict(color="#38bdf8"),
            hovertemplate="%{x|%Y-%m-%d}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=10, b=20),
        height=220,
        showlegend=False,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Alerts", gridcolor="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_controls(
    risk_df: pd.DataFrame,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], float, bool, object]:
    """Render the analyst controls (date range + risk threshold) and an about box."""
    st.sidebar.markdown('<div class="sidebar-title">Analyst controls</div>', unsafe_allow_html=True)

    earliest, latest = available_day_bounds(risk_df)
    start_date = earliest.date() if earliest is not None else None
    end_date = latest.date() if latest is not None else None

    selected = st.sidebar.date_input(
        "Date range",
        value=(start_date, end_date) if start_date and end_date else None,
        min_value=start_date,
        max_value=end_date,
        help="Limit the view to user-days within this date range.",
    )
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        start_sel, end_sel = selected
    elif isinstance(selected, (tuple, list)) and len(selected) == 1:
        start_sel = end_sel = selected[0]
    else:
        start_sel = end_sel = selected
    date_range = (pd.Timestamp(start_sel), pd.Timestamp(end_sel))

    if "daily_risk" in risk_df.columns and not risk_df.empty:
        max_risk = float(risk_df["daily_risk"].max())
    else:
        max_risk = 100.0
    slider_max = max(1.0, round(max_risk, 1))
    threshold = st.sidebar.slider(
        "Risk threshold (daily risk)",
        min_value=0.0,
        max_value=slider_max,
        value=0.0,
        step=0.1,
        help="Only user-days with daily risk at or above this value count as alerts.",
    )

    st.sidebar.markdown('<div class="sidebar-title">Replay</div>', unsafe_allow_html=True)
    replay = st.sidebar.checkbox(
        "Replay mode",
        value=False,
        help="Step through time and watch risk build up day by day - a replay of the recorded data, not a live feed.",
    )
    as_of = None
    if replay:
        days = (
            sorted(pd.to_datetime(risk_df["day"]).dt.normalize().unique())
            if "day" in risk_df.columns
            else []
        )
        day_opts = [pd.Timestamp(d).date() for d in days]
        if day_opts:
            as_of = st.sidebar.select_slider(
                "As of day",
                options=day_opts,
                value=day_opts[-1],
                format_func=lambda d: d.isoformat(),
            )

    st.sidebar.markdown(
        """
        <div class="sidebar-title">About</div>
        <div class="about-box">
          Each user-day is scored by a <b>self z-score</b>, a <b>peer z-score</b>,
          and an <b>Isolation Forest</b>, fused into a 0-100 daily risk. Weekly
          risk is the trailing-7-day sum. Ground-truth labels are used only for
          offline evaluation and are never shown here.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return date_range, threshold, replay, as_of


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """Compose the landing overview and the live dashboard from the risk scores."""
    st.set_page_config(
        page_title="UEBA Insider Threat Detection",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    config = _load_config_safe()

    risk_df, source_label = resolve_data_source(config)
    if risk_df is None:
        st.warning(
            RISK_PARQUET_MISSING_MESSAGE
            + "  Or drop CERT-schema CSVs into the sidebar uploader to score your own logs."
        )
        st.stop()
    if risk_df.empty:
        st.info("No rows to show yet. Upload CERT CSVs or run the pipeline to build risk scores.")
        st.stop()

    date_range, threshold, replay, as_of = render_controls(risk_df)
    in_range = _apply_date_range(risk_df, date_range)
    if replay and as_of is not None and not in_range.empty:
        _cutoff = pd.Timestamp(as_of).normalize()
        in_range = in_range[pd.to_datetime(in_range["day"]).dt.normalize() <= _cutoff]

    total_users = int(in_range["user"].nunique()) if not in_range.empty else 0
    total_days = int(len(in_range))
    alert_df = filter_by_threshold(in_range, threshold)
    n_alert_days = int(len(alert_df))
    n_flagged_users = int(alert_df["user"].nunique()) if not alert_df.empty else 0
    ranking_all = rank_users(in_range)
    peak_weekly = float(ranking_all["weekly_risk"].iloc[0]) if not ranking_all.empty else 0.0
    peak_user = str(ranking_all["user"].iloc[0]) if not ranking_all.empty else "-"

    render_header(config)
    st.markdown(
        f'<div class="source-pill">Showing: <b>{source_label}</b></div>',
        unsafe_allow_html=True,
    )
    if replay and as_of is not None:
        st.caption(
            f"Replay: state as of {pd.Timestamp(as_of).date().isoformat()} "
            "- a replay of recorded data, not a live feed."
        )

    tab_overview, tab_dashboard = st.tabs(["Overview", "Live dashboard"])

    with tab_overview:
        render_overview(config)

    with tab_dashboard:
        render_metric_cards(
            total_users=total_users,
            total_days=total_days,
            n_alert_days=n_alert_days,
            n_flagged_users=n_flagged_users,
            peak_weekly=peak_weekly,
            peak_user=peak_user,
            threshold=threshold,
        )
        col_left, col_right = st.columns([1.7, 1.0], gap="large")
        with col_left:
            render_top_users(in_range, threshold)
        with col_right:
            render_top_triggers(alert_df)
        render_drilldown(in_range, threshold)
        render_alerts_per_day(in_range, threshold)


if __name__ == "__main__":
    main()
