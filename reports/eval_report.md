# UEBA Detection Quality — Evaluation Report

This report grades the insider-threat detector against ground-truth labels and compares three scoring variants side by side. A plain correct-rate metric is deliberately omitted (Requirement 7.7): insider user-days are a tiny minority, so such a metric would look high while revealing nothing about whether insiders are actually caught. The metrics below are all ranking- and detection-focused instead.

**Data source:** real CMU/SEI CERT r4.2 dataset.

**Evaluated data:** 1000 users, 330452 user-days, 70 insider users, 966 insider user-days.

## Variant comparison

Each variant is ranked for the user-level metrics by the same trailing-7 aggregation (per-user maximum of a 7-day rolling sum of that variant's user-day score), so the three columns are directly comparable. PR-AUC is measured at the user-day level.

| Variant | recall@20 | recall@50 | precision@20 | pr_auc |
| --- | --- | --- | --- | --- |
| Z-score only | 0.143 | 0.200 | 0.500 | 0.042 |
| IsolationForest only | 0.071 | 0.157 | 0.250 | 0.007 |
| Fused (self + iso + peer) | 0.057 | 0.186 | 0.200 | 0.023 |

## Plots

![Detection metrics by variant](eval_metrics_bar.png)

![User-day precision-recall curves](eval_pr_curves.png)

## Limitations

These results are produced on the **real CMU/SEI CERT r4.2 dataset**. r4.2 is the "dense needles" release: it interleaves an unusually high volume of injected malicious activity compared with a real deployment, so insiders are easier to separate here than they would be in production, where malicious behaviour is far rarer and subtler. The detector is **unsupervised and ranking-based**: it surfaces anomalous user-days for an analyst to review rather than issuing automated verdicts, and the ground-truth labels are used only to score the ranking, never for training or feature construction. Results depend on a stable baseline period; users with insufficient early history are excluded from scoring. As the variant comparison shows, the IsolationForest is a weaker standalone signal than the self z-score on this data. Metrics also degrade gracefully (reported as `n/a`) when a run has no positive labels.
