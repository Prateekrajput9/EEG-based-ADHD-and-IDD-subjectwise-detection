"""Splitters.

Subject-wise (honest): every window of a child is on one side of the split.
    'loso' : leave-one-subject-out (DS-1, 14 children)
    'sgkf' : StratifiedGroupKFold, shuffled with a seed (DS-2, 121 children)
    'gkf'  : GroupKFold, unshuffled (only to reproduce the mid-semester DS-2 number)

Segment-wise (leaky, the paper's protocol): StratifiedKFold over windows, so
windows of the same child land in train and test. Kept only to measure the gap.
"""
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     StratifiedGroupKFold, StratifiedKFold)

Split = Tuple[np.ndarray, np.ndarray]


class LeakageError(AssertionError):
    """Raised when a subject appears in both the training and test side of a fold."""


def assert_no_subject_leakage(train_idx, test_idx, groups) -> None:
    groups = np.asarray(groups)
    shared = set(groups[train_idx]) & set(groups[test_idx])
    if shared:
        raise LeakageError(f"{len(shared)} subject(s) in both train and test, "
                           f"e.g. {sorted(shared)[:5]}")


def subject_splits(y, groups, scheme: str = 'auto', n_splits: int = 10,
                   seed: int = 42) -> List[Split]:
    """Subject-wise folds; each one is checked for leakage before it is returned.

    scheme='auto' picks LOSO when there are at most 20 subjects, else 'sgkf'.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    n_sub = len(np.unique(groups))
    if scheme == 'auto':
        scheme = 'loso' if n_sub <= 20 else 'sgkf'
    if scheme == 'loso':
        splitter = LeaveOneGroupOut()
    elif scheme == 'sgkf':
        splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_sub), shuffle=True,
                                        random_state=seed)
    elif scheme == 'gkf':
        splitter = GroupKFold(n_splits=min(n_splits, n_sub))
    else:
        raise ValueError(f"unknown subject split scheme {scheme!r}")
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for tr, te in splits:
        assert_no_subject_leakage(tr, te, groups)
    return splits


def segment_splits(y, n_splits: int = 10, seed: int = 42) -> List[Split]:
    """LEAKY on purpose: windows of one child appear in both train and test."""
    y = np.asarray(y)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed).split(np.zeros(len(y)), y))


def describe(scheme: str, n_splits: int, seed: int) -> dict:
    return {'scheme': scheme, 'n_splits': n_splits if scheme != 'loso' else 'n_subjects',
            'seed': seed}
