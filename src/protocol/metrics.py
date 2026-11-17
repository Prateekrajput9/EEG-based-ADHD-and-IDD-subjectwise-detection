"""Window-level and child-level metrics, with bootstrap CIs over children.

Labels: 1 = disorder (IDD or ADHD), 0 = control. `score` is P(disorder) or any
monotone score; it is used for ROC-AUC and to break majority-vote ties.

Child-level prediction = majority vote over the child's window predictions.
A tie (exactly half the windows) is broken by the mean score (>= 0.5 -> 1);
the number of ties is always reported.

CIs: percentile bootstrap that resamples CHILDREN (with replacement, separately
within each class so both classes are always present). For window metrics a
resampled child brings all of its windows. Windows of one child are not
independent, so resampling windows would give CIs that are far too narrow.
"""
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata


def _auc(y, s) -> float:
    """ROC-AUC via the rank-sum formula (ties get average rank). NaN if one class."""
    y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _basic(y, yp) -> Dict[str, float]:
    y, yp = np.asarray(y), np.asarray(yp)
    tp = int(((y == 1) & (yp == 1)).sum()); tn = int(((y == 0) & (yp == 0)).sum())
    fp = int(((y == 0) & (yp == 1)).sum()); fn = int(((y == 1) & (yp == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(sens) else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    f1_0 = 2 * prec0 * spec / (prec0 + spec) if (prec0 + spec) and not np.isnan(spec) else 0.0
    return {
        'accuracy': (tp + tn) / len(y),
        'balanced_accuracy': float(np.nanmean([sens, spec])),
        'sensitivity': sens, 'specificity': spec,
        'f1': f1, 'f1_macro': (f1 + f1_0) / 2,
        'confusion_matrix': [[tn, fp], [fn, tp]],   # rows = true 0/1, cols = pred 0/1
    }


def window_metrics(y, y_pred, score) -> Dict:
    m = _basic(y, y_pred)
    m['roc_auc'] = _auc(y, score)
    m['n'] = int(len(y))
    return m


def child_table(y, y_pred, score, groups) -> List[Dict]:
    y, y_pred, score, groups = map(np.asarray, (y, y_pred, score, groups))
    rows = []
    for g in np.unique(groups):
        m = groups == g
        frac = float(np.mean(y_pred[m] == 1))
        mean_s = float(np.mean(score[m]))
        tie = frac == 0.5
        vote = int(mean_s >= 0.5) if tie else int(frac > 0.5)
        label = int(np.bincount(y[m]).argmax())
        rows.append({'subject': str(g), 'label': label, 'n_windows': int(m.sum()),
                     'window_accuracy': float(np.mean(y_pred[m] == y[m])),
                     'fraction_predicted_disorder': frac, 'mean_score': mean_s,
                     'vote': vote, 'vote_tie': bool(tie), 'correct': int(vote == label)})
    return rows


def child_metrics(rows: List[Dict]) -> Dict:
    y = np.array([r['label'] for r in rows]); v = np.array([r['vote'] for r in rows])
    s = np.array([r['mean_score'] for r in rows])
    m = _basic(y, v)
    m['roc_auc'] = _auc(y, s)          # ranks children by mean window score
    m['n'] = int(len(rows))
    m['n_correct'] = int((y == v).sum())
    m['n_vote_ties'] = int(sum(r['vote_tie'] for r in rows))
    return m


CI_METRICS = ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'sensitivity', 'specificity')


def bootstrap_ci(y, y_pred, score, groups, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Dict:
    """Stratified (by child label) percentile bootstrap over children."""
    y, y_pred, score, groups = map(np.asarray, (y, y_pred, score, groups))
    subs = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in subs}
    lab = np.array([int(np.bincount(y[idx[g]]).argmax()) for g in subs])
    rows = child_table(y, y_pred, score, groups)
    c_vote = np.array([r['vote'] for r in rows]); c_score = np.array([r['mean_score'] for r in rows])
    pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
    rng = np.random.default_rng(seed)
    w_res = {k: [] for k in CI_METRICS}; c_res = {k: [] for k in CI_METRICS}
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        wi = np.concatenate([idx[subs[i]] for i in pick])
        wm = _basic(y[wi], y_pred[wi]); wm['roc_auc'] = _auc(y[wi], score[wi])
        cm = _basic(lab[pick], c_vote[pick]); cm['roc_auc'] = _auc(lab[pick], c_score[pick])
        for k in CI_METRICS:
            w_res[k].append(wm[k]); c_res[k].append(cm[k])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    ci = lambda v: [float(x) for x in np.nanpercentile(np.array(v, dtype=float), q)]
    return {'method': f'percentile bootstrap over children, stratified by class, '
                      f'{n_boot} resamples, seed {seed}',
            'level': 1 - alpha,
            'window': {k: ci(w_res[k]) for k in CI_METRICS},
            'child': {k: ci(c_res[k]) for k in CI_METRICS}}


def summarize(y, y_pred, score, groups, n_boot: int = 2000, seed: int = 42) -> Dict:
    rows = child_table(y, y_pred, score, groups)
    return {'window': window_metrics(y, y_pred, score),
            'child': child_metrics(rows),
            'ci95': bootstrap_ci(y, y_pred, score, groups, n_boot=n_boot, seed=seed),
            'per_child': rows}
