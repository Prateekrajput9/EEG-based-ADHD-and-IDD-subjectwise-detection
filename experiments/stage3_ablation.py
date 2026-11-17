"""Stage 3: ablations under the honest subject-wise protocol.

Parts (python -m experiments.stage3_ablation --part <name>):
  ladder_features  decomposition ladder features: raw -> fixed bands -> VMD -> MVMD
                   (SMVMD = existing cache); same 9 features + Eq. 7 integration
  ladder           each rung x {paper Table II KNN + mRMR, nested logistic regression}
  mrmr             subject-wise accuracy vs number of mRMR features (SMVMD, Table II KNN)
  window           window length {2.5, 5, 10 s} at 50% overlap and overlap {0, 50, 75%}
                   at 5 s; spectral features + nested logistic regression
  condition        DS-1 train on one condition, test on the other (LOSO children)
  summary          results/stage3/summary.{md,json}

Everything reads cached windows/features; outputs go to results/stage3/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import decompositions as dec
from src.protocol import runner, summarize, subject_splits, cross_validate
from src.protocol.results import save_result, jsonable, DS2_CONFOUND
from src.protocol.selectors import CachedMRMR
from src.utils import load_config, set_random_seed, setup_logger
from src.windows import load_windows

OUT = Path('results/stage3')
CACHE = Path('results/cache/ladder')
EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
RUNGS = ['raw', 'bands', 'vmd', 'mvmd', 'smvmd']
logger = setup_logger('Stage3', 'results/logs/stage3_ablation.log')


def alpha_for(cfg, exp):
    return cfg['ds1']['smvmd']['alpha'] if exp.startswith('DS1') else cfg['ds2']['smvmd']['alpha']


def _batch(Wb, rung, alpha):
    return np.vstack([dec.window_vector(w, rung, alpha) for w in Wb])


def ladder_matrix(cfg, exp, rung):
    """Feature matrix for one rung, row-aligned with the SMVMD cache."""
    if rung == 'smvmd':
        d = np.load(f'results/features/{exp}_features.npz'); return d['X']
    if rung == 'raw':
        return np.load(f'results/features/RAW_{exp}_features.npz')['X']
    if exp == 'DS1_RestMusic':   # rows = per child, music windows then rest windows
        _, _, g, m = load_windows(cfg, exp)
        parts = {c: (ladder_matrix(cfg, e, rung), load_windows(cfg, e)[2])
                 for c, e in (('music', 'DS1_Music'), ('rest', 'DS1_Rest'))}
        rows = []
        for sid in dict.fromkeys(g):
            for c in ('music', 'rest'):
                Xc, gc = parts[c]; rows.append(Xc[gc == sid])
        return np.vstack(rows)
    W, y, g, m = load_windows(cfg, exp)
    path = CACHE / f'{exp}_{rung}_{m["key"]}_K5.npy'
    if path.exists():
        return np.load(path)
    t0 = time.perf_counter()
    batches = np.array_split(np.arange(len(W)), max(1, len(W) // 16))
    X = np.vstack(Parallel(n_jobs=-1)(delayed(_batch)(W[b], rung, alpha_for(cfg, exp)) for b in batches))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    CACHE.mkdir(parents=True, exist_ok=True); np.save(path, X)
    logger.info(f'{exp} {rung}: {X.shape} in {time.perf_counter() - t0:.0f}s -> {path}')
    return X


def part_ladder_features(cfg, seed):
    for exp in ('DS1_Rest', 'DS1_Music', 'DS2', 'DS1_RestMusic'):
        for rung in ('bands', 'vmd', 'mvmd'):
            ladder_matrix(cfg, exp, rung)


def knn_tableii(cfg, exp):
    from experiments.stage1_eval import table_ii_model
    return table_ii_model(cfg, exp)


def lr_model():
    return (Pipeline([('scale', StandardScaler()),
                      ('clf', LogisticRegression(class_weight='balanced', max_iter=3000))]),
            {'clf__C': [0.01, 0.1, 1.0]})


def part_ladder(cfg, seed):
    for exp in EXPS:
        d = np.load(f'results/features/{exp}_features.npz')
        y, g = d['y'].astype(int), d['groups'].astype(str)
        for rung in RUNGS:
            X = ladder_matrix(cfg, exp, rung)
            knn, kdesc = knn_tableii(cfg, exp)
            runner.evaluate(OUT / 'ladder', exp, f'{rung}_knnTableII', knn, X, y, g,
                            {'name': 'KNN (paper Table II) + mRMR', 'features': f'{rung} + 9 feats + Eq.7',
                             **kdesc}, seed=seed, logger=logger)
            lr, grid = lr_model()
            runner.evaluate(OUT / 'ladder', exp, f'{rung}_lr', lr, X, y, g,
                            {'name': 'Logistic regression (nested C)', 'features': f'{rung} + 9 feats + Eq.7'},
                            seed=seed, grid=grid, logger=logger)


def part_mrmr(cfg, seed):
    """Table II KNN with n mRMR features; rankings reuse Stage 1's per-fold cache."""
    out = {}
    for exp in ('DS1_Rest', 'DS1_Music', 'DS1_RestMusic'):
        d = np.load(f'results/features/{exp}_features.npz')
        X, y, g = d['X'], d['y'].astype(int), d['groups'].astype(str)
        model, desc = knn_tableii(cfg, exp)
        rows = []
        for n in (5, 10, 20, 40, 60, 80, 100, 126):
            m = model.set_params(mrmr__n_features=n)
            r = cross_validate(m, X, y, g, subject_splits(y, g, 'loso'))
            s = summarize(y, r['y_pred'], r['score'], g, n_boot=1000, seed=seed)
            rows.append({'n_features': n, 'window_accuracy': s['window']['accuracy'],
                         'window_accuracy_ci': s['ci95']['window']['accuracy'],
                         'window_balanced_accuracy': s['window']['balanced_accuracy'],
                         'child_accuracy': s['child']['accuracy'], 'child_correct': s['child']['n_correct'],
                         'child_accuracy_ci': s['ci95']['child']['accuracy'], 'runtime_sec': r['runtime_sec']})
            logger.info(f'[mrmr {exp}] n={n}: window {rows[-1]["window_accuracy"]*100:.2f}% '
                        f'children {rows[-1]["child_correct"]}/14')
        out[exp] = {'model': desc, 'protocol': 'LOSO', 'paper_n_features': desc['n_features'], 'curve': rows}
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / 'mrmr_size_curve.json').write_text(json.dumps(jsonable(out), indent=1), encoding='utf-8')


def continuous_from_windows(W, g, start):
    """Rebuild each recording from its 50%-overlap windows (exact: windows are slices)."""
    rec = {}
    for sid in dict.fromkeys(g):
        idx = np.flatnonzero(g == sid); idx = idx[np.argsort(start[idx], kind='stable')]
        Ws = W[idx]; hop = int(start[idx[1]] - start[idx[0]]) if len(idx) > 1 else W.shape[2]
        x = np.concatenate([w[:, :hop] for w in Ws[:-1]] + [Ws[-1]], axis=1)
        rec[sid] = x
    return rec


def resegment(rec, labels, win, ovl, fs=128):
    n, step = int(win * fs), int(win * fs * (1 - ovl))
    W, Y, G = [], [], []
    for sid, x in rec.items():
        for s in range(0, x.shape[1] - n + 1, step):
            W.append(x[:, s:s + n]); Y.append(labels[sid]); G.append(sid)
    return np.stack(W).astype(np.float32), np.array(Y), np.array(G)


def part_window(cfg, seed):
    from src import features_classic as fc
    for exp in ('DS1_Rest', 'DS1_Music', 'DS2'):      # single-condition settings only
        W, y, g, m = load_windows(cfg, exp)
        rec = continuous_from_windows(W, g, m['start'])
        labels = {s: int(y[g == s][0]) for s in rec}
        for win, ovl in ((2.5, 0.5), (5, 0.5), (10, 0.5), (5, 0.0), (5, 0.75)):
            name = f'spectral_lr_win{win:g}s_ovl{int(ovl*100)}'
            if runner.exists(OUT / 'window', exp, name):
                continue
            Wn, yn, gn = resegment(rec, labels, win, ovl)
            X = fc.compute(Wn, 'spectral', f'{m["key"]}_w{win}_o{ovl}', exp)
            lr, grid = lr_model()
            runner.evaluate(OUT / 'window', exp, name, lr, X, yn, gn,
                            {'name': 'Logistic regression (nested C)', 'features': 'spectral',
                             'window_sec': win, 'overlap': ovl, 'n_windows': len(yn)},
                            seed=seed, grid=grid, logger=logger)


def part_condition(cfg, seed):
    """Train on condition A windows of the training children, test on condition B
    windows of the held-out child (LOSO). Spectral + LR (C=0.1 fixed) and SMVMD Table II KNN
    of the training condition."""
    from src import features_classic as fc
    W, y, g, m = load_windows(cfg, 'DS1_RestMusic')
    cond = m['condition']
    feats = {'spectral_lr': fc.compute(W, 'spectral', m['key'], 'DS1_RestMusic'),
             'smvmd_knnTableII': np.load('results/features/DS1_RestMusic_features.npz')['X']}
    out = {}
    for fname, X in feats.items():
        for a, b in (('rest', 'music'), ('music', 'rest'), ('rest', 'rest'), ('music', 'music')):
            if fname == 'spectral_lr':
                model = Pipeline([('scale', StandardScaler()),
                                  ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000))])
            else:
                model, _ = knn_tableii(cfg, 'DS1_Rest' if a == 'rest' else 'DS1_Music')
            splits = []
            for tr, te in subject_splits(y, g, 'loso'):
                splits.append((tr[cond[tr] == a], te[cond[te] == b]))
            keep = cond == b
            # only condition-B windows are tested, so run the folds directly
            # (cross_validate requires every window to be in some test fold)
            from sklearn.base import clone
            from src.protocol.cv import _score_of
            from src.protocol.splits import assert_no_subject_leakage
            yp = np.full(len(y), -1); sc = np.full(len(y), np.nan)
            for tr, te in splits:
                assert_no_subject_leakage(tr, te, g)
                mdl = clone(model).fit(X[tr], y[tr]); yp[te] = mdl.predict(X[te]); sc[te] = _score_of(mdl, X[te])
            s = summarize(y[keep], yp[keep], sc[keep], g[keep], n_boot=2000, seed=seed)
            out[f'{fname}|train_{a}|test_{b}'] = {'window': s['window'], 'child': s['child'], 'ci95': s['ci95']}
            logger.info(f'[condition {fname}] train {a} -> test {b}: window {s["window"]["accuracy"]*100:.2f}% '
                        f'children {s["child"]["n_correct"]}/14')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cross_condition.json').write_text(json.dumps(jsonable(
        {'protocol': 'LOSO; train on condition A of 13 children, test on condition B of the held-out child',
         'results': out}), indent=1), encoding='utf-8')


def part_summary(cfg, seed):
    from src.protocol.tables import collect, write
    for sub in ('ladder', 'window'):
        rows = collect([str(OUT / sub)])
        if rows:
            write(rows, OUT / f'{sub}_table', f'Stage 3: {sub} ablation (subject-wise)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['all', 'ladder_features', 'ladder', 'mrmr',
                                                       'window', 'condition', 'summary'])
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml'); seed = int(cfg.get('random_seed', 42)); set_random_seed(seed)
    parts = (['ladder_features', 'ladder', 'mrmr', 'window', 'condition', 'summary']
             if a.part == 'all' else [a.part])
    for p in parts:
        t0 = time.perf_counter(); logger.info(f'===== part {p} =====')
        globals()[f'part_{p}'](cfg, seed)
        logger.info(f'===== part {p} done in {time.perf_counter() - t0:.0f}s =====')


if __name__ == '__main__':
    main()
