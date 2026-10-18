"""Property-based tests for the Evaluation_Module detection metrics.

Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics match
reference computation.

These tests pin ``recall_at_k``, ``precision_at_k`` (user-level) and ``pr_auc``
(user-day level) to independently written reference definitions, so the numbers
mean exactly what a researcher expects (Requirements 7.2, 7.3). Each property
runs at least 100 Hypothesis examples. No temporary files are needed, so the
function-scoped ``tmp_path`` fixture is deliberately avoided (it trips a
Hypothesis health check when combined with ``@given``).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score

from ueba.evaluation import pr_auc, precision_at_k, recall_at_k

# Tiny tolerance for float comparisons against the reference computation.
_FLOAT_TOL = 1e-9


# ---------------------------------------------------------------------------
# Property 19 — user-level recall@k / precision@k
# ---------------------------------------------------------------------------


@st.composite
def _ranking_cases(draw):
    """Draw ``(ranking, insiders, k)`` for the user-level metrics.

    ``ranking`` is a permuted list of distinct user ids drawn from one id
    namespace. ``insiders`` mixes a sample of ids that *are* in the ranking with
    a few ids from a disjoint namespace that are *not* in the ranking (so the
    partial-insider case — insiders the detector never surfaced — is exercised).
    ``k`` is drawn to include the reported cutoffs 20 and 50 as well as values
    both below and above ``len(ranking)``.
    """
    size = draw(st.integers(min_value=0, max_value=60))
    ranking = [f"R{i:04d}" for i in range(size)]
    if ranking:
        ranking = list(draw(st.permutations(ranking)))
        insider_from_ranking = draw(
            st.lists(st.sampled_from(ranking), unique=True, max_size=size)
        )
    else:
        insider_from_ranking = []

    # Extra insiders from a disjoint namespace: real positives not in the rank.
    n_extra = draw(st.integers(min_value=0, max_value=10))
    extra = [f"X{i:04d}" for i in range(n_extra)]

    insiders = set(insider_from_ranking) | set(extra)
    # Keep the insider set non-empty so the ratio is defined; the empty-label
    # path is covered by a dedicated non-Hypothesis test below.
    if not insiders:
        insiders = {"X9999"}

    k = draw(
        st.one_of(
            st.sampled_from([20, 50]),
            st.integers(min_value=1, max_value=80),
        )
    )
    return ranking, insiders, k


@settings(max_examples=200, deadline=None)
@given(case=_ranking_cases())
def test_recall_and_precision_match_reference(case):
    """Property 19 (user level): recall@k and precision@k match the reference.

    Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics
    match reference computation. Validates Requirements 7.2, 7.3.

    For any ranking of user ids and any insider set, both metrics must equal
    their reference definitions computed independently here::

        top         = set(ranking[:k])
        recall@k    = |top ∩ insiders| / |insiders|
        precision@k = |top ∩ insiders| / k

    and both must lie in [0, 1] whenever the insider set is non-empty.
    """
    ranking, insiders, k = case

    top = set(ranking[:k])
    hits = len(top & insiders)
    ref_recall = hits / len(insiders)
    ref_precision = hits / k

    got_recall = recall_at_k(ranking, insiders, k)
    got_precision = precision_at_k(ranking, insiders, k)

    assert math.isclose(got_recall, ref_recall, abs_tol=_FLOAT_TOL, rel_tol=0.0), (
        f"recall@{k} {got_recall} != reference {ref_recall} "
        f"(hits={hits}, insiders={len(insiders)})"
    )
    assert math.isclose(
        got_precision, ref_precision, abs_tol=_FLOAT_TOL, rel_tol=0.0
    ), (
        f"precision@{k} {got_precision} != reference {ref_precision} "
        f"(hits={hits}, k={k})"
    )

    # Both metrics are genuine fractions in the unit interval.
    assert 0.0 <= got_recall <= 1.0, f"recall@{k} {got_recall} out of [0, 1]"
    assert 0.0 <= got_precision <= 1.0, f"precision@{k} {got_precision} out of [0, 1]"


@settings(max_examples=100, deadline=None)
@given(
    size=st.integers(min_value=1, max_value=30),
    k=st.sampled_from([20, 50]),
)
def test_recall_perfect_when_all_insiders_ranked_first(size, k):
    """Property 19 (user level): all insiders in the top-k gives recall 1.0.

    Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics
    match reference computation. Validates Requirements 7.2.

    A sanity anchor at the reported cutoffs k in {20, 50}: when every insider
    sits within the first ``k`` ranked users, recall@k is exactly 1.0 and
    equals the reference ``|insiders| / |insiders|``.
    """
    # Place all insiders inside the top-k window (size <= 30 but k >= 20, so
    # insiders are capped at k to stay within the window).
    n_insiders = min(size, k)
    ranking = [f"R{i:04d}" for i in range(max(size, k))]
    insiders = {f"R{i:04d}" for i in range(n_insiders)}

    assert recall_at_k(ranking, insiders, k) == pytest.approx(1.0)


def test_empty_insiders_label_path():
    """Property 19 (user level): the empty-label path degrades gracefully.

    Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics
    match reference computation. Validates Requirements 7.2.

    With no insider labels, recall@k is undefined and returns NaN (with a
    warning) while precision@k returns 0.0 (with a warning), per the reference
    graceful-degradation rule.
    """
    ranking = [f"R{i:04d}" for i in range(25)]

    with pytest.warns(RuntimeWarning):
        recall = recall_at_k(ranking, set(), 20)
    assert math.isnan(recall)

    with pytest.warns(RuntimeWarning):
        precision = precision_at_k(ranking, set(), 20)
    assert precision == 0.0


# ---------------------------------------------------------------------------
# Property 19 — user-day-level PR-AUC
# ---------------------------------------------------------------------------


@st.composite
def _pr_auc_cases(draw):
    """Draw aligned ``(scores, labels)`` with at least one positive label.

    ``labels`` are 0/1 user-day labels and ``scores`` are finite floats aligned
    one-to-one; at least one positive is guaranteed so PR-AUC is defined.
    """
    n = draw(st.integers(min_value=2, max_value=60))
    labels = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
    if 1 not in labels:
        labels[draw(st.integers(min_value=0, max_value=n - 1))] = 1
    scores = draw(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1e6,
                max_value=1e6,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return scores, labels


@settings(max_examples=150, deadline=None)
@given(case=_pr_auc_cases())
def test_pr_auc_matches_reference(case):
    """Property 19 (user-day level): PR-AUC matches the reference computation.

    Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics
    match reference computation. Validates Requirements 7.3.

    For any aligned user-day scores and 0/1 labels with at least one positive,
    ``pr_auc`` must equal ``average_precision_score(labels, scores)`` — the
    reference estimator — and lie within [0, 1].
    """
    scores, labels = case

    reference = float(average_precision_score(labels, scores))
    got = pr_auc(scores, labels)

    assert math.isclose(got, reference, abs_tol=_FLOAT_TOL, rel_tol=1e-9), (
        f"pr_auc {got} != reference average precision {reference}"
    )
    assert 0.0 <= got <= 1.0, f"pr_auc {got} out of [0, 1]"


def test_pr_auc_no_positive_labels_returns_nan():
    """Property 19 (user-day level): no positives degrades to NaN.

    Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics
    match reference computation. Validates Requirements 7.3.

    With no positive user-day labels the PR curve is undefined, so ``pr_auc``
    returns NaN after a warning rather than raising.
    """
    with pytest.warns(RuntimeWarning):
        result = pr_auc([0.1, 0.9, 0.5, 0.2], [0, 0, 0, 0])
    assert math.isnan(result)


def test_pr_auc_length_mismatch_raises():
    """Property 19 (user-day level): misaligned inputs raise ValueError.

    Feature: ueba-insider-threat-detection, Property 19: Evaluation metrics
    match reference computation. Validates Requirements 7.3.

    ``scores`` and ``labels`` must be aligned one-to-one; unequal lengths are a
    caller error and must raise ``ValueError`` rather than silently truncate.
    """
    with pytest.raises(ValueError):
        pr_auc([0.1, 0.2, 0.3], [1, 0])
