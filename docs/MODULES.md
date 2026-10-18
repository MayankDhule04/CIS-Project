# Module Guide

A plain-English explanation of every module in the UEBA Insider Threat
Detection system, in the order data flows through them. Each source file also
carries the same explanation as a docstring at the top of the file; this page
collects them in one place for review and for the viva.

Data flows top to bottom:

```
profile_data -> generate_synthetic -> ingestion -> features -> baselines
             -> models -> scoring -> evaluation -> dashboard
```

## Configuration

### `src/ueba/config.py` — typed config loader
The single place the whole system reads its settings from. It opens
`config.yaml` once, checks the important values make sense (the three fusion
weights must sum to 1.0, and every logical column name must be present in the
`column_map`), and hands back a read-only `Config` object. Every other module
asks this object for paths, the working-hours window, weekend days,
site-category lists, scoring weights, baseline-window bounds, the subset size,
and the column map — so nothing is hard-coded anywhere else.

## Scripts

### `scripts/profile_data.py` — Data_Profiler
Inspects the raw CSV logs before any parser is written. It opens each file one
slice at a time (never loading a whole file into memory) and reports its
columns, row count, and date range, so the ingestion parsers follow the data's
real shape instead of guesses. The directory it scans comes from `config.yaml`,
never a hard-coded path.

### `scripts/generate_synthetic.py` — Synthetic_Generator
A stand-in data factory. The real CERT r4.2 dataset is a ~20 GB download; this
makes small, realistic event logs in the same shape (`logon`, `device`, `http`,
`email` CSVs) plus `roles.csv` and `labels.csv`, so the whole pipeline can run,
be tested, and be demoed without the raw files. Normal days have genuine
day-to-day variation; `plant_anomaly` injects a deliberate spike (USB burst,
after-hours logons, web upload, job hunting, or email exfiltration) large enough
that the detector reliably catches it. `labels.csv` is written for evaluation
only.

### `scripts/run_eval.py` — single-command evaluation runner
One command (`python scripts/run_eval.py`) that puts `src/` on the import path
and runs the Evaluation_Module end to end: it reads `risk.parquet` (building it
from the synthetic pipeline if it is absent), compares the three scoring
variants, and writes `reports/eval_report.md` plus its plots.

## Pipeline

### `src/ueba/ingestion/` — Ingestion_Pipeline
Reads every raw event CSV a chunk at a time and boils them down to one row per
`(user, calendar_day)` in `user_day.parquet`. No raw file is ever loaded whole
into memory. The subset helpers restrict that output to the working Subset — all
labeled insiders plus a reproducible random sample of normal users — and drop
rows with a null user identifier.

### `src/ueba/features/` — Feature_Extractor
Turns each user-day into the 15-column behavioral feature table: the logon
family (logon count, after-hours logons, weekend logons, distinct machines), the
USB / removable-media family (connects, after-hours connects, files copied), the
web family (job-site visits, upload/leak-site visits, uploads, distinct domains,
total requests), and the email family (sent, external recipients, attachments).
After-hours, weekend, and site-category classifications all come from `config`.

### `src/ueba/baselines/` — Baseline_Builder
Learns what "normal" looks like. It picks each user's early-history Baseline
Period (first 60–90 days, minimum 30), then summarizes each feature as a mean
and standard deviation for the user (self baseline) and for their job-role group
(peer baseline). It uses in-period user-days only; short-history users are
flagged insufficient so scoring skips them, and zero-variance features get a
standard deviation of 0 so the downstream z-score is defined as 0 rather than
dividing by zero.

### `src/ueba/models/` — Isolation_Model
An unsupervised scikit-learn `IsolationForest` trained only on baseline-period
user-days (on or before each user's Baseline_Cutoff), with features scaled to
0–1. The fitted forest, its scaler, and the feature-column order travel together
as one bundle saved to `isolation_model.joblib`, so a reloaded model scores
identically. It returns raw anomaly scores; turning those into a 0–1 score and
fusing them is the Scoring_Engine's job.

### `src/ueba/scoring/statistical.py` — Statistical_Scorer
Compares one user-day to a baseline as per-feature z-scores, caps each absolute
z at 5, and sets the score to the mean of the top-3 capped z-scores scaled to
0–1 — once against the user's own baseline (self score) and once against the
role group (peer score). It records the top-3 contributing features and writes
`scores_stat.parquet`.

### `src/ueba/scoring/engine.py` — Scoring_Engine
Fuses the three signals. It scales the raw IsolationForest score to 0–1, then
computes `daily_risk = 100 * (0.5*self + 0.3*iso + 0.2*peer)` (weights read from
config), a `weekly_risk` that is the trailing-7-day sum per user, and a
Reason_Record of the top-3 contributing features (value, baseline mean,
z-score). It writes everything to `risk.parquet`.

## Evaluation

### `src/ueba/evaluation/metrics.py` — detection metrics
The system's single, deliberate entry point for ground-truth labels. It computes
recall@20, recall@50, and precision@20 at the user level, and PR-AUC at the
user-day level, and provides the helpers that load labels and shape
`risk.parquet` into the ranking and aligned score/label vectors the metrics
consume. Plain accuracy is deliberately never computed.

### `src/ueba/evaluation/report.py` — variant comparison + report
Grades the z-score-only, IsolationForest-only, and fused variants side by side
and writes `reports/eval_report.md` plus static matplotlib plots. `run_evaluation`
is the single-command entry point that builds or loads `risk.parquet`, compares
the variants, and writes the report.

## Dashboard

### `src/ueba/dashboard/logic.py` — pure dashboard logic
Deliberately free of any `import streamlit`. Every function here is ordinary
logic over a pandas DataFrame — finding `risk.parquet`, loading it (with an
actionable error when it is missing), ranking users by peak weekly risk,
filtering user-days by a threshold, building a user's daily timeline, finding a
user's peak day, counting alerts per day, and listing a user's top reasons — so
the dashboard's behavior can be unit- and property-tested headlessly.

### `app.py` — Streamlit dashboard
The analyst's screen and a thin rendering shell on top of the pure logic above.
It reads the pre-computed `risk.parquet`, shows users ranked by weekly risk with
a date control and a risk-threshold control, and (per user) a daily-risk
timeline, a why-flagged panel, and an alerts-per-day view. The risk file is
cached so repeated control interactions stay interactive. If `risk.parquet` is
missing it shows a clear message instead of crashing.
