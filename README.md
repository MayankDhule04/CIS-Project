# UEBA Insider Threat Detection

An offline User and Entity Behavior Analytics (UEBA) system that detects insider
threats in the CMU/SEI CERT Insider Threat Test Dataset release 4.2. It learns
each user's normal behavior during a baseline period, scores later days against
that baseline, ranks users by risk, explains why each user was flagged, and
measures detection quality against the dataset's ground-truth labels.

The system runs entirely offline: no cloud services and no runtime network
calls. Everything in this guide works on synthetic data, so you can clone the
repo and see the dashboard without the 20 GB raw dataset.

## What it does

1. **Profiles** the raw CSV logs so parsers follow the real data shape.
2. **Ingests** the event logs in chunks and aggregates them to one row per
   user per calendar day.
3. **Extracts** 15 behavioral features per user-day (logon, USB/removable
   media, web, and email activity).
4. **Baselines** each user's first 60–90 days (self baseline) and their role
   group (peer baseline).
5. **Scores** every later user-day three ways — statistical self z-score, peer
   z-score, and an IsolationForest anomaly score — and **fuses** them into a
   0–100 daily risk and a trailing-7-day weekly risk, with a top-3 "why
   flagged" reason record.
6. **Evaluates** detection quality against ground-truth labels (recall@20,
   recall@50, precision@20, and user-day PR-AUC), comparing the z-only,
   IsolationForest-only, and fused variants.
7. **Presents** everything in an interactive Streamlit dashboard: ranked users,
   per-user risk timelines, the why-flagged panel, and an alerts-per-day view.

Ground-truth labels are used for evaluation only. Nothing in ingestion,
feature extraction, baselines, or model training ever reads them.

## Tech stack

- **Python 3.11** (also runs on 3.13).
- **pandas** + **pyarrow** / **duckdb** for chunked reads and Parquet artifacts.
- **scikit-learn** `IsolationForest` for unsupervised anomaly detection.
- **Streamlit** + **Plotly** for the dashboard.
- **PyYAML** for configuration, **joblib** for model persistence,
  **matplotlib** for report plots.
- **pytest** + **Hypothesis** for unit and property tests.

Exact pinned versions live in `requirements.txt`.

## Project structure

```
CIS/
├── config.yaml          # all paths, working-hours window, site lists, weights
├── requirements.txt     # pinned dependencies
├── app.py               # Streamlit dashboard entry point
├── scripts/
│   ├── profile_data.py      # Data_Profiler: inspect raw CSVs
│   ├── generate_synthetic.py# Synthetic_Generator: CERT-shaped logs + labels
│   ├── run_eval.py          # single-command: build risk + write report
│   ├── prepare_real_data.py # build roles.csv/labels.csv for the real dataset
│   └── live_watch.py        # near-real-time: score new days as they arrive
├── src/ueba/
│   ├── config.py            # typed, validated config loader
│   ├── ingestion/           # chunked read -> one row per user-day
│   ├── features/            # 15 per-user-day behavioral features
│   ├── baselines/           # per-user + per-role mean/std baselines
│   ├── models/              # IsolationForest train/save/load/score
│   ├── scoring/             # statistical z-scores + fusion engine
│   ├── evaluation/          # metrics, variant comparison, report
│   ├── dashboard/           # pure, Streamlit-free dashboard logic
│   ├── realdata/            # adapters: LDAP->roles, answers->labels
│   └── live/                # near-real-time daily incremental scoring
├── tests/               # pytest unit + property tests per phase
├── docs/                # ARCHITECTURE.md, DEMO.md, MODULES.md
├── reports/             # eval_report.md + plots (generated)
└── data/                # raw/ + processed/  (git-ignored)
```

## Setup

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Quickstart on synthetic data

This is the full end-to-end run with no raw dataset required. Run each command
from the repository root with the virtual environment active.

```bash
# 1. Generate CERT-4.2-shaped synthetic logs (incl. roles.csv + labels.csv)
#    and plant a known insider so the demo has something to find.
python scripts/generate_synthetic.py \
    --out-dir data/raw --n-users 300 --n-days 120 --seed 42 \
    --plant USER0005:100:usb_burst

# 2. (Optional) Profile the raw CSVs to see columns, row counts, date ranges.
python scripts/profile_data.py

# 3. Build the risk scores and write the evaluation report in one command.
#    This runs ingestion -> features -> baselines -> scoring and writes
#    data/processed/risk.parquet, then reports/eval_report.md + plots.
python scripts/run_eval.py

# 4. Launch the dashboard (reads data/processed/risk.parquet).
streamlit run app.py
```

Streamlit prints a local URL (default http://localhost:8501). Open it to see
users ranked by weekly risk, select `USER0005` to inspect the planted insider's
timeline and why-flagged panel, and move the risk-threshold slider to watch the
alerts-per-day count change.

If you launch the dashboard before `risk.parquet` exists, it shows a clear
message telling you to run the pipeline first rather than crashing. Run
`python scripts/run_eval.py` and reload.

### Planting anomalies

`--plant` takes a `USER:DAY:KIND` spec and may be repeated. `DAY` is either an
integer offset from the first generated day or a `YYYY-MM-DD` date. Available
kinds:

| Kind                 | What it injects                                         |
| -------------------- | ------------------------------------------------------- |
| `usb_burst`          | A flood of USB connect/disconnect events, some after-hours |
| `after_hours_logons` | Many late-night logon/logoff pairs                      |
| `web_upload`         | A burst of visits to upload/leak sites                  |
| `job_hunt`           | A burst of job-site visits                              |
| `email_exfil`        | Emails to many external recipients with large attachments |

Example with several planted insiders:

```bash
python scripts/generate_synthetic.py --out-dir data/raw --n-users 300 \
    --n-days 120 --seed 42 \
    --plant USER0005:100:usb_burst \
    --plant USER0012:95:after_hours_logons \
    --plant USER0031:88:email_exfil
```

## Using the real CERT r4.2 dataset

The synthetic generator exists so you never need the raw dataset, but the
system runs on the real one too.

1. Download the CMU/SEI CERT Insider Threat Test Dataset release 4.2. It is a
   large download (roughly 20 GB uncompressed).
2. Place the extracted CSVs (`logon.csv`, `device.csv`, `http.csv`,
   `email.csv`, plus the role and answer/label files) into `data/raw/`. That
   directory is git-ignored, so the data never enters version control.
3. If the real column names differ from the defaults, update the `column_map`
   section of `config.yaml` — run `python scripts/profile_data.py` first to see
   the actual headers.
4. Run the same pipeline: `python scripts/run_eval.py` then
   `streamlit run app.py`.

Ingestion reads every file in chunks, so no raw CSV is ever loaded whole into
memory. Processing is restricted to the Subset (all labeled insiders plus
~300 sampled normal users) via `config.yaml`. The ground-truth labels are used
only to compute evaluation metrics — never for training or feature extraction.

## Near-real-time scoring, replay & your own data

The core pipeline is batch, but three additions make it interactive.

**Score your own CSVs.** In the dashboard sidebar, drop CERT-schema CSVs
(`logon.csv`, `device.csv`, `http.csv`, `email.csv`, optionally
`file.csv`/`roles.csv`) and the full feature -> baseline -> scoring pipeline
runs on them in-app. With nothing uploaded, the dashboard uses the built-in
`risk.parquet`. Uploads suit a sample or synthetic export; the full ~20 GB
dataset still belongs in `data/raw` and is processed via the scripts.

**Near-real-time scoring (daily micro-batch).** `scripts/live_watch.py` freezes
the baselines and model once, then scores each new day against them and appends
it to `risk.parquet`, so the dashboard reflects new days on refresh. This is
near-real-time at *daily cadence* (the granularity the features are defined at,
and how commercial UEBA tools score windows) — not per-event streaming.

```bash
python scripts/live_watch.py                 # score any new days once and exit
python scripts/live_watch.py --interval 300  # re-check every 5 minutes
```

**Replay mode.** Turn on *Replay* in the dashboard sidebar and drag the "As of
day" slider to step through the recorded timeline and watch risk build up over
time. It is a replay of recorded data, clearly labelled — not a live feed.

## Running tests

```bash
python -m pytest
```

The entire suite (unit and Hypothesis property tests) runs against
Synthetic_Generator output, so it never needs the raw dataset.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture diagrams, the
  feature table, the scoring formula, the final weights, evaluation results,
  and system limitations.
- [`docs/DEMO.md`](docs/DEMO.md) — a ~5-minute demonstration walkthrough on
  synthetic data.
- [`docs/MODULES.md`](docs/MODULES.md) — a plain-English, one-paragraph guide to
  every module in the system.

## Module guide (at a glance)

Each source module also carries a plain-English docstring at the top of its
file. Here is the one-line version; see [`docs/MODULES.md`](docs/MODULES.md) for
more detail.

| Module | In plain English |
| ------ | ---------------- |
| `src/ueba/config.py` | Reads and validates `config.yaml`, handing the rest of the system one read-only settings object. |
| `src/ueba/ingestion/` | Reads raw event CSVs in chunks and boils them down to one row per user per day. |
| `src/ueba/features/` | Turns each user-day into 15 behavioral numbers (logon, USB, web, email), using config for after-hours and site categories. |
| `src/ueba/baselines/` | Learns what "normal" looks like for each user and each role from their first 60–90 days. |
| `src/ueba/models/` | Trains, saves, loads, and scores the IsolationForest on baseline-period days only. |
| `src/ueba/scoring/statistical.py` | Compares a day to the self and peer baselines as capped z-scores and keeps the top-3 drivers. |
| `src/ueba/scoring/engine.py` | Fuses the self, isolation, and peer scores into daily/weekly risk with a why-flagged reason record. |
| `src/ueba/evaluation/metrics.py` | Grades the ranking against ground-truth labels (recall@k, precision@k, PR-AUC). |
| `src/ueba/evaluation/report.py` | Compares the three scoring variants and writes `reports/eval_report.md` plus plots. |
| `src/ueba/dashboard/logic.py` | Pure, Streamlit-free helpers for ranking, thresholding, timelines, and reasons. |
| `app.py` | The Streamlit UI that renders the dashboard on top of that logic. |
| `scripts/profile_data.py` | Inspects raw CSVs (columns, row counts, date ranges) before parsers are written. |
| `scripts/generate_synthetic.py` | Produces CERT-4.2-shaped synthetic logs, roles, and labels, with optional planted anomalies. |
| `scripts/run_eval.py` | One command that builds `risk.parquet` (if missing) and writes the evaluation report. |
| `src/ueba/realdata/` | Adapters that build `roles.csv` from LDAP and `labels.csv` from the CERT answers for the real dataset. |
| `src/ueba/live/` | Freezes the baselines/model and scores new days against them (near-real-time, daily micro-batch). |
| `scripts/prepare_real_data.py` | Builds `roles.csv`/`labels.csv` into the raw dir so the real dataset runs through the pipeline. |
| `scripts/live_watch.py` | Scores any new days and appends them to `risk.parquet` (near-real-time, once or on an interval). |
