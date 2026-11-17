"""Rule 1: no subject may appear in both train and test of any fold.

Checks (a) the splitters on the real DS-1 / DS-2 subject layouts, (b) the inner
folds of nested CV, and (c) with a spy estimator, that no fitted step ever sees
a window of the outer test children, including during hyper-parameter tuning.
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.protocol import (LeakageError, assert_no_subject_leakage, cross_validate,
                          nested_cv, segment_splits, subject_splits)


def layout(n_subjects, windows_per_subject, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f's{i:03d}' for i in range(n_subjects)], windows_per_subject)
    labels = np.repeat(np.arange(n_subjects) % 2, windows_per_subject)
    X = rng.normal(size=(len(groups), 5))
    return X, labels, groups


LAYOUTS = {'DS1_like': layout(14, 46), 'DS1_restmusic_like': layout(14, 92),
           'DS2_like': layout(121, 54)}


@pytest.mark.parametrize('name', LAYOUTS)
@pytest.mark.parametrize('scheme', ['loso', 'sgkf', 'gkf', 'auto'])
def test_subject_splits_never_share_subjects(name, scheme):
    X, y, g = LAYOUTS[name]
    splits = subject_splits(y, g, scheme=scheme, n_splits=10, seed=1)
    seen_test = []
    for tr, te in splits:
        assert not set(g[tr]) & set(g[te])
        assert set(g[tr]) | set(g[te]) == set(g)
        seen_test += list(set(g[te]))
    assert sorted(seen_test) == sorted(set(g)), 'every subject is tested exactly once'


def test_segment_split_does_leak_and_guard_catches_it():
    X, y, g = LAYOUTS['DS1_like']
    tr, te = segment_splits(y, 10, 0)[0]
    with pytest.raises(LeakageError):
        assert_no_subject_leakage(tr, te, g)
    with pytest.raises(LeakageError):
        cross_validate(_Spy(), np.c_[np.arange(len(y)), X], y, g, segment_splits(y, 10, 0))


def test_real_dataset_groups_if_cached():
    import os
    for exp in ('DS1_Rest', 'DS1_RestMusic', 'DS2'):
        path = f'results/features/{exp}_features.npz'
        if not os.path.exists(path):
            pytest.skip('feature caches not present')
        d = np.load(path)
        y, g = d['y'].astype(int), d['groups'].astype(str)
        for tr, te in subject_splits(y, g, scheme='auto', n_splits=10, seed=42):
            assert not set(g[tr]) & set(g[te])


class _Spy(BaseEstimator, ClassifierMixin):
    """Column 0 carries the row id. Records every row id it was fitted on."""
    log = []

    def __init__(self, c=1.0):
        self.c = c

    def fit(self, X, y):
        _Spy.log.append(set(X[:, 0].astype(int).tolist()))
        self.classes_ = np.array([0, 1])
        self.p_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), int(self.p_ >= 0.5))

    def predict_proba(self, X):
        return np.tile([1 - self.p_, self.p_], (len(X), 1))


@pytest.mark.parametrize('name', ['DS1_like', 'DS2_like'])
def test_no_fit_ever_sees_test_children(name):
    X, y, g = LAYOUTS[name]
    Xid = np.c_[np.arange(len(y)), X]
    outer = subject_splits(y, g, scheme='auto', n_splits=5, seed=3)
    # sequential (n_jobs=1) so the spy log is in this process
    _Spy.log = []
    cross_validate(_Spy(), Xid, y, g, outer, n_jobs=1)
    for (tr, te), seen in zip(outer, _Spy.log):
        assert not seen & set(te.tolist())
    # nested: every fit made while handling outer fold k must exclude its test rows
    for k, (tr, te) in enumerate(outer):
        _Spy.log = []
        nested_cv(_Spy(), {'c': [1.0, 2.0]}, Xid, y, g, [(tr, te)], inner_scheme='sgkf',
                  inner_n_splits=3, n_jobs=1)
        assert len(_Spy.log) == 3 * 2 + 1      # inner folds x grid + final refit
        test_rows = set(te.tolist())
        for seen in _Spy.log:
            assert not seen & test_rows


def test_inner_folds_are_subject_wise_and_inside_outer_train():
    X, y, g = LAYOUTS['DS1_like']
    for tr, te in subject_splits(y, g, 'loso'):
        inner = subject_splits(y[tr], g[tr], 'sgkf', 4, 42)
        gi = g[tr]
        for itr, ite in inner:
            assert not set(gi[itr]) & set(gi[ite])
            assert not set(gi[ite]) & set(g[te])
