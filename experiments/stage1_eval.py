"""Stage 1: SMVMD + mRMR + KNN under the unified protocol (src/protocol).

Parts (python -m experiments.stage1_eval --part <name>):
  tableii   Paper Table II KNN, unchanged hyper-parameters. Same model under the
            leaky segment-wise split and the honest subject-wise split (the gap),
            with window + child metrics and bootstrap CIs over children.
            DS-2 is also run with the mid-semester GroupKFold split so the two
            reports can be tied together.
  nested    KNN tuned by nested subject-wise CV (outer: LOSO for DS-1, seeded
            StratifiedGroupKFold(10) for DS-2; inner: StratifiedGroupKFold(4) on
            the outer training children). mRMR size, scaling, k, metric and
            weighting are all chosen inside the outer training set.
  splitvar  DS-2 Table II KNN over 20 different subject-wise fold seeds: how much
            the honest number moves just because of how children are split.
  figures   segment-vs-subject gap figure, from the saved JSON files only.

Inputs: results/features/<exp>_features.npz (SMVMD features, unchanged).
Outputs: results/stage1_eval/<exp>/<model>__<protocol>.json|.npz, figures/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.knn_model import squared_inverse_weights
from src.protocol import (cross_validate, nested_cv, save_result, segment_splits,
                          subject_splits, summarize)
from src.protocol.selectors import CachedMRMR
from src.utils import load_config, set_random_seed, setup_logger

OUT = Path('results/stage1_eval')
logger = setup_logger('Stage1Eval', 'results/logs/stage1_eval.log')

EXPS = {  # exp -> (dataset, knn config key, mRMR config key)
    'DS1_Rest': ('ds1', 'rest', 'n_features_rest'),
    'DS1_Music': ('ds1', 'music', 'n_features_music'),
    'DS1_RestMusic': ('ds1', 'rest_music', 'n_features_rest_music'),
    'DS2': ('ds2', None, 'n_features'),
}
N_BOOT = 2000


def load_features(exp):
    d = np.load(f'results/features/{exp}_features.npz')
    return d['X'], d['y'].astype(int), d['groups'].astype(str), str(d['fingerprint'])


def _weights(name):
    return squared_inverse_weights if name == 'squared_inverse' else (
        'uniform' if name in ('uniform', 'equal') else 'distance')


def table_ii_model(cfg, exp):
    """Exactly the mid-semester pipeline: [scale] -> mRMR (in-fold) -> KNN (Table II)."""
    ds, kk, nk = EXPS[exp]
    kp = cfg['ds1']['knn'][kk] if ds == 'ds1' else cfg['ds2']['knn']
    nf = cfg[ds]['mrmr'][nk]
    metric = {'cityblock': 'manhattan'}.get(kp['distance'], kp['distance'])
    pipe = Pipeline([
        ('scale', StandardScaler() if kp.get('standardize') else 'passthrough'),
        ('mrmr', CachedMRMR(n_features=nf)),
        ('knn', KNeighborsClassifier(n_neighbors=kp['n_neighbors'], metric=metric,
                                     weights=_weights(kp['weights']), n_jobs=1)),
    ])
    return pipe, {**kp, 'n_features': nf}


def tuned_model_and_grid(exp, n_total):
    pipe = Pipeline([('mrmr', CachedMRMR(n_features=n_total)),
                     ('scale', 'passthrough'),
                     ('knn', KNeighborsClassifier(n_jobs=1))])
    # DS-2: all 171 features as in the paper (an mRMR ranking of DS-2 costs ~8 min,
    # and the paper keeps every feature), so only the KNN settings are tuned.
    nfs = [n_total] if exp == 'DS2' else [10, 20, 40, 80, n_total]
    grid = {'mrmr__n_features': nfs,
            'scale': ['passthrough', StandardScaler()],
            'knn__n_neighbors': [1, 3, 5, 9, 15, 25],
            'knn__metric': ['euclidean', 'manhattan', 'cosine'],
            'knn__weights': ['uniform', 'distance', squared_inverse_weights]}
    return pipe, grid


def subject_scheme(exp):
    return 'loso' if exp.startswith('DS1') else 'sgkf'


def _save(exp, name, protocol, model_desc, res, y, g, cfg, fingerprint, seed, extra=None):
    summ = summarize(y, res['y_pred'], res['score'], g, n_boot=N_BOOT, seed=seed)
    payload = {'experiment': exp, 'model': model_desc, 'protocol': protocol,
               'n_folds': res['n_folds'], 'runtime_sec': res['runtime_sec'],
               'metrics': {'window': summ['window'], 'child': summ['child']},
               'ci95': summ['ci95'], 'per_child': summ['per_child'], **(extra or {})}
    config = {'dataset': EXPS[exp][0].upper(), 'experiment': exp,
              'feature_fingerprint': fingerprint, 'model': model_desc,
              'protocol': protocol, 'n_boot': N_BOOT}
    path = save_result(OUT / exp, f'{name}__{protocol["name"]}', payload,
                       {'y': y, 'y_pred': res['y_pred'], 'score': res['score'],
                        'fold': res['fold'], 'groups': g}, config, seed)
    w, c = summ['window'], summ['child']
    logger.info(f"[{exp} | {name} | {protocol['name']}] window acc {w['accuracy']*100:.2f}% "
                f"bal {w['balanced_accuracy']*100:.2f}% AUC {w['roc_auc']:.3f} | child "
                f"{c['n_correct']}/{c['n']} ({c['accuracy']*100:.1f}%, CI "
                f"{summ['ci95']['child']['accuracy'][0]*100:.0f}-"
                f"{summ['ci95']['child']['accuracy'][1]*100:.0f}%) [{res['runtime_sec']}s] -> {path}")
    return summ


def part_tableii(cfg, seed):
    for exp in EXPS:
        X, y, g, fp = load_features(exp)
        model, desc = table_ii_model(cfg, exp)
        desc = {'name': 'SMVMD+mRMR+KNN (paper Table II)', **desc}
        seg = cross_validate(model, X, y, g, segment_splits(y, 10, seed), leaky_ok=True)
        _save(exp, 'knn_tableII', {'name': 'segment10', 'split': 'StratifiedKFold(10) over windows',
                                   'leaky': True, 'seed': seed}, desc, seg, y, g, cfg, fp, seed)
        sch = subject_scheme(exp)
        sub = cross_validate(model, X, y, g, subject_splits(y, g, sch, 10, seed))
        _save(exp, 'knn_tableII', {'name': f'subject_{sch}', 'split': sch, 'n_splits': 10,
                                   'leaky': False, 'seed': seed}, desc, sub, y, g, cfg, fp, seed)
        if exp == 'DS2':
            old = cross_validate(model, X, y, g, subject_splits(y, g, 'gkf', 10, seed))
            _save(exp, 'knn_tableII', {'name': 'subject_gkf', 'split': 'GroupKFold(10), '
                                       'unshuffled (mid-semester split)', 'leaky': False,
                                       'seed': None}, desc, old, y, g, cfg, fp, seed)


def part_nested(cfg, seed):
    for exp in EXPS:
        sch = subject_scheme(exp)
        if (OUT / exp / f'knn_tuned__subject_{sch}.json').exists():
            logger.info(f'[{exp} | knn_tuned] exists, skipped (delete the JSON to re-run)')
            continue
        X, y, g, fp = load_features(exp)
        model, grid = tuned_model_and_grid(exp, X.shape[1])
        res = nested_cv(model, grid, X, y, g, subject_splits(y, g, sch, 10, seed),
                        inner_scheme='sgkf', inner_n_splits=4, seed=seed)
        desc = {'name': 'SMVMD+mRMR+KNN (nested-tuned)', 'grid': grid}
        _save(exp, 'knn_tuned', {'name': f'subject_{sch}', 'split': sch, 'n_splits': 10,
                                 'leaky': False, 'seed': seed,
                                 'inner': res['inner'], 'criterion': res['criterion']},
              desc, res, y, g, cfg, fp, seed,
              extra={'folds': res['folds'], 'grid_size': res['grid_size']})


def part_splitvar(cfg, seed, n_seeds=20):
    X, y, g, fp = load_features('DS2')
    model, desc = table_ii_model(cfg, 'DS2')
    rows = []
    for s in range(n_seeds):
        r = cross_validate(model, X, y, g, subject_splits(y, g, 'sgkf', 10, s))
        m = summarize(y, r['y_pred'], r['score'], g, n_boot=1, seed=s)
        rows.append({'split_seed': s, 'window_accuracy': m['window']['accuracy'],
                     'window_balanced_accuracy': m['window']['balanced_accuracy'],
                     'window_roc_auc': m['window']['roc_auc'],
                     'child_accuracy': m['child']['accuracy'], 'child_correct': m['child']['n_correct']})
    acc = np.array([r['window_accuracy'] for r in rows]); cacc = np.array([r['child_accuracy'] for r in rows])
    out = {'experiment': 'DS2', 'model': desc, 'split': 'sgkf(10)', 'seeds': list(range(n_seeds)),
           'per_seed': rows,
           'window_accuracy': {'mean': acc.mean(), 'std': acc.std(ddof=1), 'min': acc.min(), 'max': acc.max()},
           'child_accuracy': {'mean': cacc.mean(), 'std': cacc.std(ddof=1), 'min': cacc.min(), 'max': cacc.max()}}
    from src.protocol.results import DS2_CONFOUND, jsonable
    out['notes'] = [DS2_CONFOUND]
    (OUT / 'DS2').mkdir(parents=True, exist_ok=True)
    (OUT / 'DS2' / 'knn_tableII__split_seed_variance.json').write_text(
        json.dumps(jsonable(out), indent=1), encoding='utf-8')
    logger.info(f"[DS2 split variance] window acc {acc.mean()*100:.2f} +/- {acc.std(ddof=1)*100:.2f} "
                f"(range {acc.min()*100:.2f}-{acc.max()*100:.2f}) over {n_seeds} split seeds")


def part_figures(cfg, seed):
    from src.protocol.figures import gap_figure
    gap_figure(OUT, OUT / 'figures' / 'segment_vs_subject_gap')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['all', 'tableii', 'nested', 'splitvar', 'figures'])
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml')
    seed = int(cfg.get('random_seed', 42))
    set_random_seed(seed)
    parts = ['tableii', 'splitvar', 'nested', 'figures'] if a.part == 'all' else [a.part]
    for p in parts:
        t0 = time.perf_counter()
        logger.info(f'===== part {p} =====')
        {'tableii': part_tableii, 'nested': part_nested, 'splitvar': part_splitvar,
         'figures': part_figures}[p](cfg, seed)
        logger.info(f'===== part {p} done in {time.perf_counter() - t0:.0f}s =====')


if __name__ == '__main__':
    main()
