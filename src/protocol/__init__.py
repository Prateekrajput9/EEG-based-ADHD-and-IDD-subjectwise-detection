"""Unified, leakage-safe evaluation protocol used by every experiment from Stage 1 on.

    splits.py     subject-wise / segment-wise splitters and the leakage guard
    metrics.py    window- and child-level metrics, bootstrap CIs over children
    selectors.py  mRMR as a pipeline step, ranking cached on disk
    cv.py         cross-validation, nested CV, segment-vs-subject gap
    results.py    standard JSON + NPZ result files (config hash, seed, git commit)

Rule enforced everywhere: every fitted step (scaling, feature selection,
tuning, early stopping) sees only the training children of the current fold.
"""
from .splits import (LeakageError, assert_no_subject_leakage, segment_splits,
                     subject_splits)
from .metrics import summarize, window_metrics, child_table, child_metrics, bootstrap_ci
from .cv import cross_validate, nested_cv, segment_vs_subject_gap
from .results import save_result, load_result, config_hash

__all__ = [
    'LeakageError', 'assert_no_subject_leakage', 'segment_splits', 'subject_splits',
    'summarize', 'window_metrics', 'child_table', 'child_metrics', 'bootstrap_ci',
    'cross_validate', 'nested_cv', 'segment_vs_subject_gap',
    'save_result', 'load_result', 'config_hash',
]
