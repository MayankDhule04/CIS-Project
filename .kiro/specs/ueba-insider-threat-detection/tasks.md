# Implementation Plan: UEBA Insider Threat Detection

## Overview

This plan implements the UEBA system in seven sequential phases, exactly matching the design's phase-to-acceptance mapping. Each phase ends with a checkpoint requiring its unit and property tests to pass before the next phase begins. Implementation is in Python 3.11 using pandas, DuckDB/pyarrow, scikit-learn IsolationForest, Streamlit + Plotly, PyYAML, and pytest with Hypothesis for property tests.

All tests run against Synthetic_Generator output so the suite never needs the 20 GB raw dataset. Property tests use Hypothesis with a minimum of 100 iterations and are tagged `Feature: ueba-insider-threat-detection, Property {number}: {property_text}`. Tasks that require the real CERT 4.2 dataset (and therefore cannot run in CI) are marked optional with `*`.

## Tasks

- [x] 1. Phase 1 — Setup & profiling
  - [x] 1.1 Create project skeleton and packaging
    - Create `config.yaml`, `requirements.txt`, `README.md`, `app.py`, `.gitignore`, and the `scripts/`, `tests/`, `docs/`, `reports/`, `data/raw/`, `data/processed/` directories
    - Create `src/ueba/__init__.py` and the `ingestion`, `features`, `baselines`, `models`, `scoring`, `evaluation` subpackages (each with `__init__.py`)
    - Add `docs/ARCHITECTURE.md` and `docs/DEMO.md` placeholder files
    - Pin dependencies in `requirements.txt` (pandas, pyarrow, duckdb, scikit-learn, streamlit, plotly, pyyaml, pytest, hypothesis)
    - Make `.gitignore` exclude `data/`
    - _Requirements: 1.1, 1.5_

  - [x] 1.2 Implement typed config loader
    - Implement `src/ueba/config.py` with the frozen `Config` dataclass and nested `WorkingHours`, `SiteCategories`, `FusionWeights`, `BaselineConfig`, `SubsetConfig`, and `column_map` structures
    - Implement `load_config(path="config.yaml")` reading the raw data dir, processed dir, working hours, weekend days, site categories, weights, baseline bounds, subset config, and column map
    - Validate that fusion weights sum to 1.0 within tolerance and that every `column_map` logical name is present, raising clear configuration errors otherwise
    - Populate `config.yaml` with the design's default schema values
    - _Requirements: 1.2, 1.4_

  - [x] 1.3 Implement Data_Profiler
    - Implement `scripts/profile_data.py` with `profile_directory(raw_dir) -> list[FileProfile]` and the `FileProfile` dataclass (`filename`, `columns`, `row_count`, `date_range`)
    - Read each CSV in chunks and print header, row count, and date range per file
    - Read the raw data directory path from the Config_File, not a hardcoded value
    - _Requirements: 1.3, 1.4_

  - [x]* 1.4 Write Phase 1 smoke tests
    - Assert the project structure exists (packages, dirs, `app.py`, docs, reports) (1.1)
    - Assert required config keys load and resolve, including config-driven path resolution (1.2, 1.4)
    - Assert `.gitignore` excludes `data/` (1.5)
    - Assert the test suite runs green under pytest (1.6)
    - Assert no runtime network calls occur during a representative run (network-blocked run) (10.3)
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 10.3_

  - [x]* 1.5 Write Property 1 test for Data_Profiler
    - **Property 1: Data_Profiler reports match ground truth**
    - Generate arbitrary CSVs (header, rows, date column); assert reported row count, column list, and date range equal the true values
    - **Validates: Requirements 1.3**

  - [x] 1.6 Checkpoint — Phase 1 tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 2. Phase 2 — Ingestion + synthetic generator
  - [x] 2.1 Implement Synthetic_Generator
    - Implement `scripts/generate_synthetic.py` with `generate(out_dir, n_users, n_days, seed)` writing CSVs matching the CERT 4.2 schema per file type (logon, device, http, email)
    - Implement `plant_anomaly(user_id, day, kind)` injecting measurable spikes (USB burst, after-hours logons, etc.) for a user on a day
    - Ensure output is consumable downstream without the raw dataset
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 2.2 Implement chunked ingestion and daily aggregation
    - Implement `src/ueba/ingestion/` with `ingest(config) -> pd.DataFrame`: chunked read of every raw CSV (DuckDB/pyarrow), aggregate to one row per `(user, calendar_day)`, drop null-user rows, write `user_day.parquet`
    - Ensure no raw file is loaded whole into memory
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.3 Implement subset selection
    - Implement `select_subset(users, insiders, sample_size, seed)` returning all insiders plus ~`sample_size` sampled normals, driven by config
    - Restrict ingestion output to the Subset
    - _Requirements: 2.6_

  - [x]* 2.4 Write Property 6 and 7 tests for the generator
    - **Property 6: Synthetic logs match the dataset schema**
    - **Validates: Requirements 3.1**
    - **Property 7: Planted anomalies create measurable deviation**
    - **Validates: Requirements 3.2, 5.6**

  - [x]* 2.5 Write Property 2, 3, 4 tests for ingestion
    - **Property 2: Aggregation yields exactly one record per user-day**
    - **Validates: Requirements 2.2, 2.4**
    - **Property 3: Null-user events are excluded without side effects**
    - **Validates: Requirements 2.5**
    - **Property 4: Subset includes every insider**
    - **Validates: Requirements 2.6**

  - [x]* 2.6 Write Property 5 round-trip test for `user_day.parquet`
    - **Property 5: Artifact persistence round-trips**
    - **Validates: Requirements 2.3**

  - [x]* 2.7 Write ingestion example and integration tests
    - Example: assert the chunked-reader path is taken (not a whole-file load) (2.1)
    - Integration: generator → ingestion pipeline produces valid `user_day.parquet` on synthetic data (2.7, 3.3)
    - _Requirements: 2.1, 2.7, 3.3_

  - [x]* 2.8 Run ingestion against the real CERT 4.2 dataset
    - Run the full ingestion over the real 20 GB dataset Subset to confirm chunked memory behavior and row counts; optional because it cannot run in CI
    - _Requirements: 2.1, 2.6_

  - [x] 2.9 Checkpoint — Phase 2 tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Phase 3 — Features, baselines, statistical scoring
  - [x] 3.1 Implement Feature_Extractor
    - Implement `src/ueba/features/` with `extract_features(user_day_events, config)` computing all `user_day.parquet` feature columns: logon family (4.1), USB/removable-media family (4.2), web family (4.3), email family (4.4)
    - Classify after-hours activity and site categories strictly from config working-hours window and site lists (4.5)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x]* 3.2 Write Property 8 and 9 tests for features
    - **Property 8: Feature counts equal independently derived truth**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    - **Property 9: Classification follows config**
    - **Validates: Requirements 4.5**

  - [x] 3.3 Implement Baseline_Builder
    - Implement `src/ueba/baselines/` with `baseline_cutoff(user_history)` (first 60–90 days, minimum 30), `build_self_baseline(features_in_period)`, and `build_peer_baseline(features_in_period, roles)`
    - Use only in-period user-days; handle short-history users (flag insufficient baseline, exclude from scoring) and zero-variance features (z defined as 0)
    - _Requirements: 5.1, 5.2_

  - [x]* 3.4 Write Property 10 and 11 tests for baselines
    - **Property 10: Baselines depend only on in-period user-days**
    - **Validates: Requirements 5.1**
    - **Property 11: Baseline window respects bounds**
    - **Validates: Requirements 5.2**

  - [x] 3.5 Implement Statistical_Scorer
    - Implement `src/ueba/scoring/statistical.py` with `self_score(user_day, self_baseline)` and `peer_score(user_day, peer_baseline)`: per-feature z vs baseline, cap each |z| at 5, score = mean of top-3 capped z scaled to 0–1 (divide by 5)
    - Write `scores_stat.parquet` including `self_top_features` and `peer_top_features` structs
    - _Requirements: 5.3, 5.4, 5.5_

  - [x]* 3.6 Write Property 12 and Property 5 tests for statistical scoring
    - **Property 12: Statistical score invariants hold for self and peer**
    - **Validates: Requirements 5.3, 5.4**
    - **Property 5: Artifact persistence round-trips** (`scores_stat.parquet`)
    - **Validates: Requirements 5.5**

  - [x]* 3.7 Write edge-case tests for scoring
    - Edge: planted USB spike produces a z-score greater than 4 on the spiked feature (5.6)
    - Edge: zero-variance baseline feature yields z = 0; short-history user excluded
    - _Requirements: 5.6_

  - [x] 3.8 Checkpoint — Phase 3 tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Phase 4 — IsolationForest + fusion
  - [x] 4.1 Implement Isolation_Model train/save/load/score
    - Implement `src/ueba/models/` with `train(baseline_features)` fitting IsolationForest on user-days dated on or before the Baseline_Cutoff only, features scaled to 0–1
    - Implement `save(model, path)` / `load(path)` persisting to `data/processed/isolation_model.joblib`, and `score(model, user_days)`
    - Exclude any candidate record dated after the Baseline_Cutoff from training
    - _Requirements: 6.1, 6.2, 6.9_

  - [x]* 4.2 Write Property 13 and 18 and model-saved smoke tests
    - **Property 13: IsolationForest trains only on baseline-period records**
    - **Validates: Requirements 6.1, 6.2**
    - **Property 18: Saved and reloaded model score identically**
    - **Validates: Requirements 6.10**
    - Smoke: model file saved to disk after training (6.9)
    - _Requirements: 6.9, 6.10_

  - [x] 4.3 Implement Scoring_Engine fusion, reasons, and persistence
    - Implement `src/ueba/scoring/engine.py` with `scale_isolation(raw_scores) -> 0..1`, `daily_risk(self_s, iso_s, peer_s, weights) = 100*(w_self*self + w_iso*iso + w_peer*peer)`, `weekly_risk(daily_series)` (trailing-7 sum per user), and `reason_record(user_day, baselines)` (top-3 features with value, baseline mean, z-score)
    - Read fusion weights from config; write `risk.parquet` with daily_risk, weekly_risk, scores, and reasons
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x]* 4.4 Write Property 14, 15, 16, 17 and Property 5 tests for fusion
    - **Property 14: Isolation score is bounded**
    - **Validates: Requirements 6.3**
    - **Property 15: Daily risk equals the configured weighted fusion**
    - **Validates: Requirements 6.4, 6.5**
    - **Property 16: Weekly risk equals the trailing-7 daily sum**
    - **Validates: Requirements 6.6**
    - **Property 17: Reason records are the true top-3 contributors**
    - **Validates: Requirements 6.7**
    - **Property 5: Artifact persistence round-trips** (`risk.parquet`)
    - **Validates: Requirements 6.8**

  - [x] 4.5 Checkpoint — Phase 4 tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Phase 5 — Evaluation
  - [x] 5.1 Implement evaluation metrics
    - Implement `src/ueba/evaluation/` with `recall_at_k(ranking, labels, k)` and `precision_at_k(ranking, labels, k)` at user level (k = 20, 50) and `pr_auc(scores, labels)` at user-day level
    - Degrade gracefully on empty label sets (0/NaN with warning); never compute plain accuracy
    - Consume Ground_Truth_Label data here only, never in features/baselines/models
    - _Requirements: 7.1, 7.2, 7.3, 7.7_

  - [x]* 5.2 Write Property 19 test for evaluation metrics
    - **Property 19: Evaluation metrics match reference computation**
    - **Validates: Requirements 7.2, 7.3**

  - [x] 5.3 Implement variant comparison and report generation
    - Implement `compare_variants(...)` reporting z-only, iso-only, and fused side by side
    - Implement `write_report(results, out="reports/eval_report.md")` producing markdown plus Plotly/Matplotlib plots, runnable from a single command
    - _Requirements: 7.4, 7.5, 7.6_

  - [x]* 5.4 Write evaluation example and integration tests
    - Example: three variants present, no plain accuracy in report, report file and plots exist (7.4, 7.5, 7.7)
    - Example: labels never reach features/model/baselines (label isolation) (7.1)
    - Integration: single command generates `reports/eval_report.md` and plots (7.6)
    - _Requirements: 7.1, 7.4, 7.5, 7.6, 7.7_

  - [x]* 5.5 Run evaluation on the real CERT 4.2 dataset
    - Produce the evaluation report and variant comparison on the real labeled dataset to record final results; optional because it cannot run in CI
    - _Requirements: 7.2, 7.3, 7.4_

  - [x] 5.6 Checkpoint — Phase 5 tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Phase 6 — Dashboard
  - [x] 6.1 Implement dashboard ranking and controls
    - Implement `app.py` (Streamlit) reading pre-computed `risk.parquet`; display users ranked by Weekly_Risk, with a date control and a risk-threshold control
    - Show an actionable message if `risk.parquet` is missing instead of crashing
    - _Requirements: 8.1, 8.2_

  - [x] 6.2 Implement user drill-down and alerts-per-day
    - On user selection, show a daily-risk timeline and a why-flagged panel (each contributing feature value vs baseline); show top contributing reasons for insiders
    - Add an alerts-per-day view counting user-days at or above the threshold
    - Keep render-time compute light so the Subset becomes interactive within 10 seconds
    - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x]* 6.3 Write Property 20, 21, 22 tests for the dashboard
    - **Property 20: Users are ranked by descending weekly risk**
    - **Validates: Requirements 8.1**
    - **Property 21: Drill-down data matches stored data**
    - **Validates: Requirements 8.3, 8.4, 8.7**
    - **Property 22: Alerts-per-day matches the threshold definition**
    - **Validates: Requirements 8.5**

  - [x]* 6.4 Write dashboard example and integration tests
    - Example: date and risk-threshold controls render (8.2)
    - Integration: dashboard becomes interactive within 10 seconds on the Subset (8.6)
    - _Requirements: 8.2, 8.6_

  - [x] 6.5 Checkpoint — Phase 6 tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Phase 7 — Docs & demo polish
  - [x] 7.1 Write ARCHITECTURE.md
    - Fill `docs/ARCHITECTURE.md` with Mermaid diagrams, the feature table, the scoring formula, the final scoring weights, the evaluation results, and the system limitations
    - _Requirements: 9.1_

  - [x] 7.2 Write README.md and DEMO.md with per-module explanations
    - Write `README.md` with setup and run instructions sufficient to clone and view the dashboard on synthetic data
    - Write `docs/DEMO.md` describing how to run the demonstration
    - Add a plain-English explanation accompanying each module
    - _Requirements: 9.2, 9.3, 9.4_

  - [x]* 7.3 Write Phase 7 smoke and integration tests
    - Smoke: required docs and their sections are present; per-phase tests exist and the phase plan is covered (9.1–9.4, 10.1, 10.2)
    - Integration: a fresh clone following README launches the dashboard on synthetic data (9.5)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2_

  - [x] 7.4 Final checkpoint — all phases' tests pass
    - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional. These are test sub-tasks and the two real-dataset runs (2.8, 5.5) that cannot execute in CI because they require the 20 GB CERT 4.2 dataset; everything else runs on Synthetic_Generator output.
- Core implementation sub-tasks (no `*`) MUST be implemented; optional test sub-tasks follow each implementation to catch errors early.
- Each phase ends with a checkpoint; the next phase does not begin until the current phase's unit and property tests pass.
- Property tests use Hypothesis (≥ 100 iterations) and are tagged `Feature: ueba-insider-threat-detection, Property {number}: {property_text}`.
- Property 5 (artifact round-trip) recurs in Phases 2, 3, and 4 because three distinct artifacts (`user_day.parquet`, `scores_stat.parquet`, `risk.parquet`) are produced.
- Each task references specific requirement clauses for traceability; all 22 correctness properties are covered by property-test sub-tasks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4", "1.5", "2.1"] },
    { "id": 3, "tasks": ["2.2", "2.4"] },
    { "id": 4, "tasks": ["2.3", "2.5", "2.6", "2.7", "2.8"] },
    { "id": 5, "tasks": ["3.1"] },
    { "id": 6, "tasks": ["3.2", "3.3"] },
    { "id": 7, "tasks": ["3.4", "3.5"] },
    { "id": 8, "tasks": ["3.6", "3.7", "4.1"] },
    { "id": 9, "tasks": ["4.2", "4.3"] },
    { "id": 10, "tasks": ["4.4", "5.1"] },
    { "id": 11, "tasks": ["5.2", "5.3"] },
    { "id": 12, "tasks": ["5.4", "5.5", "6.1"] },
    { "id": 13, "tasks": ["6.2"] },
    { "id": 14, "tasks": ["6.3", "6.4", "7.1"] },
    { "id": 15, "tasks": ["7.2"] },
    { "id": 16, "tasks": ["7.3"] }
  ]
}
```
