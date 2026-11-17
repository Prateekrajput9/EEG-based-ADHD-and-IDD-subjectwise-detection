"""Cross-validation drivers.

cross_validate  : one fresh clone of the model per fold, fitted on that fold's
                  training windows only; returns out-of-fold predictions.
nested_cv       : outer subject-wise folds for testing; inside each outer
                  training set, an inner subject-wise CV picks hyper-parameters.
                  The outer test children are never seen during tuning.
segment_vs_subject_gap : the same model under the leaky segment-wise split and
                  the honest subject-wise split, for the "gap" figure.

Every fold passes through assert_no_subject_leakage (except the deliberately
leaky segment-wise split, which is labelled as such).
"""
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from .metrics import summarize
from .splits import assert_no_subject_leakage, segment_splits, subject_splits


def _score_of(model, X) -> np.ndarray:
    """P(class 1) if available, else a decision score squashed to (0, 1)."""
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        classes = list(getattr(model, 'classes_', [0, 1]))
        return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def _fit(model, X, y, groups, pass_groups):
    if pass_groups:
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def _fit_predict(model, X, y, groups, tr, te, pass_groups):
    m = _fit(clone(model), X[tr], y[tr], groups[tr], pass_groups)
    return m.predict(X[te]), _score_of(m, X[te])


def cross_validate(model, X, y, groups, splits, n_jobs: int = -1,
                   leaky_ok: bool = False, pass_groups: bool = False) -> Dict[str, Any]:
    """Out-of-fold predictions for every window.

    leaky_ok=True must be passed explicitly to run a split that shares subjects
    between train and test (only the segment-wise comparison does this).
    """
    y, groups = np.asarray(y), np.asarray(groups)
    if not leaky_ok:
        for tr, te in splits:
            assert_no_subject_leakage(tr, te, groups)
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_fit_predict)(model, X, y, groups, tr, te, pass_groups) for tr, te in splits)
    y_pred = np.full(len(y), -1); score = np.full(len(y), np.nan); fold = np.full(len(y), -1)
    for k, ((tr, te), (p, s)) in enumerate(zip(splits, out)):
        y_pred[te], score[te], fold[te] = p, s, k
    assert (fold >= 0).all(), 'some windows were never in a test fold'
    return {'y_pred': y_pred, 'score': score, 'fold': fold,
            'runtime_sec': round(time.perf_counter() - t0, 2), 'n_folds': len(splits)}


def _inner_select(model, grid, X, y, groups, inner_splits, criterion, pass_groups):
    """Pooled out-of-fold inner score for every grid point; returns best params + table."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    preds = [np.full(len(y), -1) for _ in grid]
    scores = [np.full(len(y), np.nan) for _ in grid]
    for tr, te in inner_splits:
        assert_no_subject_leakage(tr, te, groups)
        for j, params in enumerate(grid):
            m = _fit(clone(model).set_params(**params), X[tr], y[tr], groups[tr], pass_groups)
            preds[j][te] = m.predict(X[te]); scores[j][te] = _score_of(m, X[te])
    table = []
    for j, params in enumerate(grid):
        if criterion == 'roc_auc':
            v = roc_auc_score(y, scores[j])
        else:
            v = balanced_accuracy_score(y, preds[j])
        table.append(v)
    best = int(np.argmax(table))     # ties -> first grid point (grids list simpler models first)
    return grid[best], float(table[best]), table


def _outer_fold(model, grid, X, y, groups, tr, te, inner_scheme, inner_n_splits, seed,
                criterion, pass_groups):
    t0 = time.perf_counter()
    inner = subject_splits(y[tr], groups[tr], scheme=inner_scheme,
                           n_splits=inner_n_splits, seed=seed)
    best, inner_score, table = _inner_select(model, grid, X[tr], y[tr], groups[tr],
                                             inner, criterion, pass_groups)
    m = _fit(clone(model).set_params(**best), X[tr], y[tr], groups[tr], pass_groups)
    return (m.predict(X[te]), _score_of(m, X[te]), best, inner_score, table,
            round(time.perf_counter() - t0, 1))


def nested_cv(model, param_grid, X, y, groups, outer_splits, inner_scheme: str = 'sgkf',
              inner_n_splits: int = 4, seed: int = 42, criterion: str = 'balanced_accuracy',
              n_jobs: int = -1, pass_groups: bool = False) -> Dict[str, Any]:
    """Nested subject-wise CV. Tuning criterion is pooled inner window-level
    balanced accuracy (or ROC-AUC); the outer test children never enter tuning."""
    y, groups = np.asarray(y), np.asarray(groups)
    grid = list(ParameterGrid(param_grid))
    for tr, te in outer_splits:
        assert_no_subject_leakage(tr, te, groups)
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_outer_fold)(model, grid, X, y, groups, tr, te, inner_scheme,
                             inner_n_splits, seed, criterion, pass_groups)
        for tr, te in outer_splits)
    y_pred = np.full(len(y), -1); score = np.full(len(y), np.nan); fold = np.full(len(y), -1)
    folds = []
    for k, ((tr, te), (p, s, best, isc, table, rt)) in enumerate(zip(outer_splits, out)):
        y_pred[te], score[te], fold[te] = p, s, k
        folds.append({'fold': k, 'test_subjects': sorted(set(groups[te].tolist())),
                      'chosen_params': best, 'inner_score': isc, 'runtime_sec': rt})
    return {'y_pred': y_pred, 'score': score, 'fold': fold, 'folds': folds,
            'grid_size': len(grid), 'criterion': criterion,
            'inner': {'scheme': inner_scheme, 'n_splits': inner_n_splits, 'seed': seed},
            'runtime_sec': round(time.perf_counter() - t0, 2), 'n_folds': len(outer_splits)}


def segment_vs_subject_gap(model, X, y, groups, subject_scheme: str = 'auto',
                           n_splits: int = 10, seed: int = 42, n_boot: int = 2000,
                           n_jobs: int = -1, pass_groups: bool = False) -> Dict[str, Any]:
    """Same model, two splits: leaky segment-wise vs honest subject-wise."""
    y, groups = np.asarray(y), np.asarray(groups)
    seg = cross_validate(model, X, y, groups, segment_splits(y, n_splits, seed),
                         n_jobs=n_jobs, leaky_ok=True, pass_groups=pass_groups)
    sub = cross_validate(model, X, y, groups,
                         subject_splits(y, groups, subject_scheme, n_splits, seed),
                         n_jobs=n_jobs, pass_groups=pass_groups)
    s_seg = summarize(y, seg['y_pred'], seg['score'], groups, n_boot, seed)
    s_sub = summarize(y, sub['y_pred'], sub['score'], groups, n_boot, seed)
    return {'segment_wise': {**seg, 'summary': s_seg},
            'subject_wise': {**sub, 'summary': s_sub},
            'gap_window_accuracy': s_seg['window']['accuracy'] - s_sub['window']['accuracy'],
            'gap_window_balanced_accuracy': (s_seg['window']['balanced_accuracy']
                                             - s_sub['window']['balanced_accuracy'])}
