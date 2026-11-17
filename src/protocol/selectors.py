"""mRMR as a scikit-learn pipeline step, refitted on every training fold.

The full mRMR ranking of a training set costs about a minute (DS-1) to eight
minutes (DS-2), and nested CV fits the same training set many times (once per
hyper-parameter combination). The ranking is therefore cached, keyed by a hash
of the exact training matrix and labels it was computed from. A cache hit can
only happen for an identical training set, so caching cannot leak anything.
The greedy ranking is deterministic, so its first n entries are exactly the
ranking that would be computed for n features; one full ranking serves every n.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

CACHE_DIR = Path(os.environ.get('MRMR_CACHE_DIR', 'results/cache/mrmr'))
_MEM = {}


def _key(X, y, max_samples_redundancy, random_state) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    h.update(f'{X.shape}|{max_samples_redundancy}|{random_state}|v1'.encode())
    return h.hexdigest()


def full_mrmr_ranking(X, y, max_samples_redundancy=2000, random_state=42) -> np.ndarray:
    key = _key(X, y, max_samples_redundancy, random_state)
    if key in _MEM:
        return _MEM[key]
    path = CACHE_DIR / f'{key}.npy'
    if path.exists():
        r = np.load(path)
    else:
        import logging
        from src.mrmr import compute_mrmr_ranking
        lvl = logging.getLogger('src.mrmr').level
        logging.getLogger('src.mrmr').setLevel(logging.WARNING)
        try:
            r = compute_mrmr_ranking(X, y, n_features=X.shape[1],
                                     max_samples_redundancy=max_samples_redundancy,
                                     random_seed=random_state)
        finally:
            logging.getLogger('src.mrmr').setLevel(lvl)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp.npy')
        np.save(tmp, r)
        os.replace(tmp, path)
    _MEM[key] = r
    return r


class CachedMRMR(BaseEstimator, TransformerMixin):
    """Keep the top `n_features` mRMR-ranked columns (all columns if n >= p)."""

    def __init__(self, n_features=80, max_samples_redundancy=2000, random_state=42):
        self.n_features = n_features
        self.max_samples_redundancy = max_samples_redundancy
        self.random_state = random_state

    def fit(self, X, y):
        p = X.shape[1]
        if self.n_features is None or self.n_features >= p:
            self.idx_ = np.arange(p)
        else:
            self.idx_ = full_mrmr_ranking(X, y, self.max_samples_redundancy,
                                          self.random_state)[:self.n_features]
        return self

    def transform(self, X):
        return X[:, self.idx_]
