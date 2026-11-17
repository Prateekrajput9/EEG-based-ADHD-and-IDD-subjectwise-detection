import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from src.protocol.metrics import bootstrap_ci, child_metrics, child_table, window_metrics


def _data(seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat([f's{i}' for i in range(10)], 20)
    y = np.repeat(np.arange(10) % 2, 20)
    s = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.3, len(y)), 0, 1)
    return y, (s >= 0.5).astype(int), s, g


def test_window_metrics_match_sklearn():
    y, yp, s, _ = _data()
    m = window_metrics(y, yp, s)
    assert np.isclose(m['balanced_accuracy'], balanced_accuracy_score(y, yp))
    assert np.isclose(m['f1'], f1_score(y, yp))
    assert np.isclose(m['f1_macro'], f1_score(y, yp, average='macro'))
    assert np.isclose(m['roc_auc'], roc_auc_score(y, s))


def test_vote_tie_broken_by_mean_score():
    y = np.array([1, 1, 1, 1]); g = np.array(['a'] * 4)
    rows = child_table(y, np.array([1, 1, 0, 0]), np.array([.9, .8, .4, .4]), g)
    assert rows[0]['vote_tie'] and rows[0]['vote'] == 1
    rows = child_table(y, np.array([1, 1, 0, 0]), np.array([.6, .6, .1, .1]), g)
    assert rows[0]['vote'] == 0
    assert child_metrics(rows)['n_vote_ties'] == 1


def test_bootstrap_ci_brackets_point_estimate():
    y, yp, s, g = _data(1)
    ci = bootstrap_ci(y, yp, s, g, n_boot=300, seed=0)
    w = window_metrics(y, yp, s)
    lo, hi = ci['window']['accuracy']
    assert lo <= w['accuracy'] <= hi
    c = child_metrics(child_table(y, yp, s, g))
    lo, hi = ci['child']['accuracy']
    assert lo <= c['accuracy'] <= hi
    assert bootstrap_ci(y, yp, s, g, n_boot=300, seed=0) == ci      # seeded
