# Requirements Document

## Introduction

This document specifies the requirements for a User and Entity Behavior Analytics (UEBA) system that detects insider threats using the CMU/SEI CERT Insider Threat Test Dataset release 4.2. The system learns each user's normal behavior during a baseline period, scores later days against that baseline, ranks users by risk, explains why each user was flagged, and measures detection quality against the dataset's ground-truth labels.

The deliverable is a university project comprising a running demonstration, the source code, and architecture documentation. The system runs entirely offline with no cloud services and no runtime network calls. It is built in Python 3.11 using pandas with DuckDB or pyarrow for chunked processing, scikit-learn IsolationForest for anomaly detection, a Streamlit and Plotly dashboard, pytest for testing, and PyYAML for configuration. The work is organized into seven sequential phases, each independently tested before the next begins.

## Glossary

- **UEBA_System**: The complete user and entity behavior analytics system, including ingestion, feature extraction, baseline computation, scoring, evaluation, and dashboard components.
- **Data_Profiler**: The component implemented in `scripts/profile_data.py` that inspects each raw CSV file and reports its structure before any parser is written.
- **Ingestion_Pipeline**: The component that reads raw event CSV files in chunks and aggregates them into one row per user per day.
- **Feature_Extractor**: The component that computes per-user-per-day behavioral features from ingested events.
- **Baseline_Builder**: The component that computes per-user statistical baselines from baseline-period user-days.
- **Statistical_Scorer**: The component that computes self z-score and peer z-score anomaly scores.
- **Isolation_Model**: The scikit-learn IsolationForest model trained on baseline-period user-days.
- **Scoring_Engine**: The component that fuses self, peer, and isolation scores into daily and weekly risk scores.
- **Evaluation_Module**: The component that computes detection quality metrics and generates the evaluation report.
- **Dashboard**: The Streamlit application defined in `app.py` that presents ranked users, drill-downs, and alerts.
- **Synthetic_Generator**: The component that generates synthetic event logs matching the dataset schema for tests and demonstrations.
- **Config_File**: The `config.yaml` file holding all configurable paths, working-hours definitions, site-category lists, and scoring weights.
- **User_Day**: A single aggregated record representing one user's behavior on one calendar day.
- **Baseline_Period**: The first 60 to 90 days of a user's activity, with a minimum of 30 days, used to learn normal behavior.
- **Baseline_Cutoff**: The date that separates a user's baseline period from the later scoring period.
- **Ground_Truth_Label**: A dataset-provided label identifying known insider-threat user-days, used only for evaluation.
- **Self_Score**: An anomaly score comparing a user-day against that same user's own baseline.
- **Peer_Score**: An anomaly score comparing a user-day against the user's role group baseline.
- **Isolation_Score**: An anomaly score produced by the Isolation_Model, scaled to the range 0 to 1.
- **Daily_Risk**: The fused daily risk score on a 0 to 100 scale.
- **Weekly_Risk**: The sum of the last seven Daily_Risk values for a user.
- **Reason_Record**: The stored set of top-3 contributing features for a user-day, each with its value, baseline mean, and z-score.
- **Subset**: The working data subset comprising all labeled insider users plus approximately 300 randomly selected normal users.

## Requirements

### Requirement 1: Project Setup and Data Profiling

**User Story:** As a developer, I want the project environment and data profiling established first, so that parsers are derived from real data structure rather than assumptions.

#### Acceptance Criteria

1. THE UEBA_System SHALL provide a project structure containing `config.yaml`, `requirements.txt`, `README.md`, a `scripts/` directory, an `src/ueba/` package with `ingestion`, `features`, `baselines`, `models`, `scoring`, and `evaluation` subpackages, `app.py`, a `tests/` directory, `docs/ARCHITECTURE.md`, `docs/DEMO.md`, and a `reports/` directory.
2. THE Config_File SHALL define the raw data directory path, working-hours definitions, site-category lists, and scoring weights.
3. WHEN the Data_Profiler is run against the raw data directory, THE Data_Profiler SHALL print the column header, row count, and date range for each CSV file.
4. THE UEBA_System SHALL read the raw data directory path from the Config_File rather than from a hardcoded value.
5. THE UEBA_System SHALL exclude the raw data directory contents from version control.
6. WHEN pytest is run, THE UEBA_System SHALL execute the test suite successfully.

### Requirement 2: Event Ingestion and Daily Aggregation

**User Story:** As a developer, I want raw events ingested and aggregated into per-user-per-day records, so that behavioral features can be computed from a compact dataset.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline processes a raw CSV file, THE Ingestion_Pipeline SHALL read the file in chunks rather than loading the entire file into memory at once.
2. THE Ingestion_Pipeline SHALL aggregate raw events into one User_Day record per user per calendar day.
3. THE Ingestion_Pipeline SHALL write aggregated User_Day records to a Parquet file named `user_day.parquet`.
4. THE Ingestion_Pipeline SHALL produce a User_Day row count equal to the number of users multiplied by their active days.
5. THE Ingestion_Pipeline SHALL exclude records that contain a null user identifier from `user_day.parquet`.
6. THE Ingestion_Pipeline SHALL process the Subset consisting of all labeled insider users plus approximately 300 randomly selected normal users.
7. WHEN unit tests are run against synthetic data produced by the Synthetic_Generator, THE Ingestion_Pipeline SHALL pass the unit tests.

### Requirement 3: Synthetic Log Generation

**User Story:** As a developer, I want a synthetic log generator with the dataset schema, so that tests and demonstrations run without the 20 GB raw dataset.

#### Acceptance Criteria

1. THE Synthetic_Generator SHALL generate event logs using the same schema as the CERT release 4.2 dataset CSV files.
2. WHEN the Synthetic_Generator is requested to plant an anomaly, THE Synthetic_Generator SHALL produce events that create a measurable deviation for a specified user on a specified day.
3. THE Synthetic_Generator SHALL produce data usable by the Ingestion_Pipeline, Feature_Extractor, Scoring_Engine, and Dashboard without the raw dataset.

### Requirement 4: Behavioral Feature Extraction

**User Story:** As an analyst, I want per-user-per-day behavioral features computed, so that each user-day is represented by measurable activity signals.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL compute, for each User_Day, the logon count, the after-hours logon count, the weekend logon count, and the count of distinct machines.
2. THE Feature_Extractor SHALL compute, for each User_Day, the USB connect count, the after-hours USB connect count, and the count of files copied to removable media.
3. THE Feature_Extractor SHALL compute, for each User_Day, the job-site visit count, the file-upload-or-leak-site visit count, the file-upload count, the distinct-domain count, and the total web request count.
4. THE Feature_Extractor SHALL compute, for each User_Day, the emails-sent count, the external-recipient count, and the email-attachment count.
5. WHERE working-hours definitions and site-category lists are specified in the Config_File, THE Feature_Extractor SHALL classify after-hours activity and site categories according to those Config_File values.

### Requirement 5: Statistical Baselines and Z-Score Scoring

**User Story:** As an analyst, I want statistical baselines and z-score scoring, so that user-days can be compared against each user's own normal behavior and against peers.

#### Acceptance Criteria

1. THE Baseline_Builder SHALL compute per-user baselines using only User_Day records within the Baseline_Period.
2. THE Baseline_Period SHALL span the first 60 to 90 days of each user's activity, with a minimum of 30 days.
3. WHEN the Statistical_Scorer computes a Self_Score for a User_Day, THE Statistical_Scorer SHALL compute each feature z-score against the user's own baseline, cap each z-score at 5, and set the Self_Score to the mean of the top-3 feature z-scores scaled to the range 0 to 1.
4. WHEN the Statistical_Scorer computes a Peer_Score for a User_Day, THE Statistical_Scorer SHALL compute each feature z-score against the user's role-group baseline, cap each z-score at 5, and set the Peer_Score to the mean of the top-3 feature z-scores scaled to the range 0 to 1.
5. THE Statistical_Scorer SHALL write statistical scores to a Parquet file named `scores_stat.parquet`.
6. WHEN a User_Day contains a planted USB spike in a test, THE Statistical_Scorer SHALL produce a z-score greater than 4 for that User_Day on the spiked feature.

### Requirement 6: IsolationForest Scoring and Risk Fusion

**User Story:** As an analyst, I want IsolationForest anomaly scoring fused with statistical scores, so that risk reflects both statistical deviation and machine-learned anomaly patterns.

#### Acceptance Criteria

1. THE Isolation_Model SHALL be trained using only User_Day records dated on or before the Baseline_Cutoff.
2. IF a candidate training record is dated after the Baseline_Cutoff, THEN THE Isolation_Model SHALL exclude that record from training.
3. THE Scoring_Engine SHALL scale the Isolation_Score to the range 0 to 1.
4. THE Scoring_Engine SHALL compute Daily_Risk as 100 multiplied by the sum of 0.5 times the Self_Score, 0.3 times the Isolation_Score, and 0.2 times the Peer_Score.
5. THE Scoring_Engine SHALL read the self, isolation, and peer fusion weights from the Config_File.
6. THE Scoring_Engine SHALL compute Weekly_Risk as the sum of the last seven Daily_Risk values for each user.
7. THE Scoring_Engine SHALL store a Reason_Record of the top-3 contributing features for each User_Day, with each feature's value, baseline mean, and z-score.
8. THE Scoring_Engine SHALL write Daily_Risk, Weekly_Risk, and Reason_Record data to a Parquet file named `risk.parquet`.
9. WHEN the Isolation_Model is trained, THE Scoring_Engine SHALL save the trained model to disk.
10. WHEN a test reloads the saved Isolation_Model, THE Scoring_Engine SHALL produce scores from the reloaded model.

### Requirement 7: Detection Quality Evaluation

**User Story:** As a researcher, I want detection quality evaluated against ground-truth labels, so that the system's effectiveness is measured and the scoring variants are compared.

#### Acceptance Criteria

1. THE Evaluation_Module SHALL use Ground_Truth_Label data only for evaluation and SHALL NOT use Ground_Truth_Label data for training or feature computation.
2. THE Evaluation_Module SHALL compute recall at 20, recall at 50, and precision at 20 at the user level.
3. THE Evaluation_Module SHALL compute the precision-recall area under the curve at the user-day level.
4. THE Evaluation_Module SHALL report metrics for the z-score-only variant, the IsolationForest-only variant, and the fused variant side by side.
5. THE Evaluation_Module SHALL write an evaluation report to `reports/eval_report.md` with accompanying plots.
6. WHEN a single command is run, THE Evaluation_Module SHALL generate `reports/eval_report.md` and its plots.
7. THE Evaluation_Module SHALL exclude plain accuracy from the evaluation report.

### Requirement 8: Interactive Risk Dashboard

**User Story:** As an analyst, I want an interactive dashboard, so that I can review ranked users, inspect individual behavior, and understand why users were flagged.

#### Acceptance Criteria

1. THE Dashboard SHALL display users ranked by Weekly_Risk.
2. THE Dashboard SHALL provide a date control and a risk-threshold control.
3. WHEN an analyst selects a user, THE Dashboard SHALL display a daily-risk timeline for that user.
4. WHEN an analyst selects a user, THE Dashboard SHALL display a why-flagged panel showing each contributing feature's value against its baseline.
5. THE Dashboard SHALL display an alerts-per-day view.
6. WHEN the Dashboard is opened on the Subset, THE Dashboard SHALL become interactive within 10 seconds.
7. WHEN an analyst selects a known insider user, THE Dashboard SHALL display the top contributing reasons for that user.

### Requirement 9: Documentation and Reproducible Demo

**User Story:** As a new contributor, I want clear documentation and a reproducible demo, so that I can run the system on synthetic data without the raw dataset.

#### Acceptance Criteria

1. THE UEBA_System SHALL provide `docs/ARCHITECTURE.md` containing Mermaid diagrams, a feature table, the scoring formula, the final scoring weights, the evaluation results, and the system limitations.
2. THE UEBA_System SHALL provide `docs/DEMO.md` describing how to run the demonstration.
3. THE UEBA_System SHALL provide a `README.md` with setup and run instructions sufficient to clone the repository and view the Dashboard on synthetic data.
4. THE UEBA_System SHALL provide a plain-English explanation accompanying each module.
5. WHEN a new contributor clones the repository and follows the `README.md` instructions, THE UEBA_System SHALL display the Dashboard on synthetic data.

### Requirement 10: Phased, Tested Delivery

**User Story:** As a developer, I want the project built in tested, sequential phases, so that each capability is verified before the next is started.

#### Acceptance Criteria

1. THE UEBA_System SHALL be organized into seven phases covering setup and profiling, ingestion, baselines and statistical scoring, IsolationForest and fusion, evaluation, dashboard, and documentation and demo polish.
2. THE UEBA_System SHALL provide unit tests for each phase.
3. THE UEBA_System SHALL operate without cloud services and without runtime network calls.
