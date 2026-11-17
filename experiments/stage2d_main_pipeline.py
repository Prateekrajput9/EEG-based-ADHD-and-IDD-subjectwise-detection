"""Stage 2d: supporting analyses for the main pipeline
   SMVMD common modes -> covariance -> Riemannian tangent space -> LightGBM.

Reads the Stage 2c covariance cache; writes results/stage2d/.

  perm     Subject-level label permutation test on DS-1 (rest, music). The 14
           people's labels are shuffled (7 IDD / 7 TDC kept), the full LOSO is
           re-run, and the real result is compared with the permutation
           distribution. Fixed configuration (all four modes, LightGBM depth 4):
           no inner CV, so each permutation is cheap; the configuration is not
           tuned on any test person.
  gap      The same fixed pipeline under window-wise (leaky) and subject-wise
           splits, DS-2 and DS-1 rest: how much would window-level leakage
           inflate this pipeline?
  importance  LightGBM gain importance, averaged over the 10 outer DS-2 folds
           (and the 14 DS-1 rest folds), mapped back to mode, channel and channel
           pair. Descriptive only.
  runtime  Time per window of each stage on this laptop (single thread):
           SMVMD, covariances, tangent space + LightGBM prediction; training time
           of one model on all DS-2 training windows; model size (trees, leaves).
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from joblib import Parallel, delayed

import stage2c_smvmd_riemann as s2c
from src.protocol import summarize
from src.protocol.cv import cross_validate
from src.protocol.splits import segment_splits, subject_splits
from src.utils import load_config, set_random_seed, setup_logger
from src.windows import load_windows

OUT = Path('results/stage2d')
BANDS = [b for b, *_ in s2c.BANDS]
logger = setup_logger('Stage2d', 'results/logs/stage2d_main_pipeline.log')


def fixed_pipe():
    p, _ = s2c.pipe((0, 1, 2, 3), 'lgbm')
    return p.set_params(clf__max_depth=4)        # the depth chosen in every DS-2 outer fold


def load(cfg, exp):
    d = np.load(s2c.cache_path(cfg, exp))
    return d['covs_smvmd'], d['y'].astype(int), d['groups'].astype(str)


def channels(cfg, exp):
    return list(cfg['ds2' if exp == 'DS2' else 'ds1']['channel_names'])


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}.json').write_text(json.dumps(obj, indent=2))
    logger.info(f'-> {OUT / name}.json')


# ------------------------------------------------------------------ permutation
def _one_run(X, y, g, splits, seed):
    r = cross_validate(fixed_pipe(), X, y, g, splits, n_jobs=1)
    s = summarize(y, r['y_pred'], r['score'], g, n_boot=2, seed=seed)   # CIs unused here
    return s['window']['balanced_accuracy'], s['child']['n_correct']


def part_perm(cfg, seed, n_jobs, n_perm):
    out = {}
    for exp in ['DS1_Rest', 'DS1_Music']:
        X, y, g = load(cfg, exp)
        splits = subject_splits(y, g, 'loso', 10, seed)
        subs = np.array(sorted(set(g)))
        lab = np.array([y[g == s][0] for s in subs])
        t0 = time.perf_counter()
        real_bal, real_n = _one_run(X, y, g, splits, seed)
        rng = np.random.default_rng(seed)
        perms = [dict(zip(subs, rng.permutation(lab))) for _ in range(n_perm)]
        res = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_one_run)(X, np.array([p[s] for s in g]), g, splits, seed) for p in perms)
        bal = np.array([r[0] for r in res]); nc = np.array([r[1] for r in res])
        n_identity = int(sum(all(p[s] == l for s, l in zip(subs, lab)) for p in perms))
        out[exp] = {
            'pipeline': 'SMVMD common modes (all four) + tangent space + LightGBM (depth 4, fixed)',
            'n_permutations': n_perm, 'permutation_unit': 'subject (7/7 kept)',
            'real_window_bal_acc': real_bal, 'real_people_correct': int(real_n), 'n_people': len(subs),
            'perm_window_bal_acc_mean': float(bal.mean()), 'perm_window_bal_acc_p95': float(np.percentile(bal, 95)),
            'perm_window_bal_acc_max': float(bal.max()),
            'perm_people_correct_mean': float(nc.mean()), 'perm_people_correct_max': int(nc.max()),
            'perm_people_correct_hist': {int(k): int(v) for k, v in zip(*np.unique(nc, return_counts=True))},
            'p_value_bal_acc': float((1 + (bal >= real_bal).sum()) / (n_perm + 1)),
            'p_value_people': float((1 + (nc >= real_n).sum()) / (n_perm + 1)),
            'n_perms_equal_to_true_labels': n_identity,
            'runtime_sec': round(time.perf_counter() - t0, 1),
            'perm_window_bal_acc': bal.round(4).tolist(), 'perm_people_correct': nc.tolist(),
        }
        logger.info(f'{exp}: real {real_bal*100:.1f}% ({real_n}/14); permuted mean {bal.mean()*100:.1f}%, '
                    f'max {bal.max()*100:.1f}%, people max {nc.max()}/14; p = {out[exp]["p_value_bal_acc"]:.4f}')
    save('permutation_ds1', out)


# ------------------------------------------------------------------ gap
def part_gap(cfg, seed, n_jobs):
    out = {}
    for exp in ['DS2', 'DS1_Rest']:
        X, y, g = load(cfg, exp)
        seg = cross_validate(fixed_pipe(), X, y, g, segment_splits(y, 10, seed), n_jobs=n_jobs, leaky_ok=True)
        sch = 'loso' if exp.startswith('DS1') else 'sgkf'
        sub = cross_validate(fixed_pipe(), X, y, g, subject_splits(y, g, sch, 10, seed), n_jobs=n_jobs)
        r = {}
        for k, v in [('window_wise_leaky', seg), ('subject_wise', sub)]:
            s = summarize(y, v['y_pred'], v['score'], g, n_boot=2000, seed=seed)
            r[k] = {'window_acc': s['window']['accuracy'], 'window_bal_acc': s['window']['balanced_accuracy'],
                    'window_auc': s['window']['roc_auc'], 'ci95_window_bal_acc': s['ci95']['window']['balanced_accuracy'],
                    'people_correct': s['child']['n_correct'], 'n_people': s['child']['n']}
        r['gap_bal_acc_points'] = 100 * (r['window_wise_leaky']['window_bal_acc'] - r['subject_wise']['window_bal_acc'])
        out[exp] = r
        logger.info(f'{exp}: window-wise {r["window_wise_leaky"]["window_bal_acc"]*100:.1f}% vs subject-wise '
                    f'{r["subject_wise"]["window_bal_acc"]*100:.1f}% (gap {r["gap_bal_acc_points"]:.1f} pts)')
    out['pipeline'] = 'SMVMD common modes (all four) + tangent space + LightGBM (depth 4, fixed)'
    save('gap_main_pipeline', out)


# ------------------------------------------------------------------ importance
def _fold_importance(X, y, tr):
    m = fixed_pipe().fit(X[tr], y[tr])
    return m.named_steps['clf'].model_.booster_.feature_importance(importance_type='gain')


def part_importance(cfg, seed, n_jobs):
    out = {}
    for exp in ['DS2', 'DS1_Rest']:
        X, y, g = load(cfg, exp)
        ch = channels(cfg, exp); C = X.shape[-1]
        assert len(ch) == C, (len(ch), C)
        sch = 'loso' if exp.startswith('DS1') else 'sgkf'
        splits = subject_splits(y, g, sch, 10, seed)
        imps = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_fold_importance)(X, y, tr) for tr, _ in splits)
        imp = np.mean(imps, 0); imp = imp / imp.sum()
        iu = np.triu_indices(C)                       # pyriemann tangent-vector order (upper triangle)
        n_per = len(iu[0]); assert len(imp) == 4 * n_per
        per_mode = {b: float(imp[k * n_per:(k + 1) * n_per].sum()) for k, b in enumerate(BANDS)}
        per_ch = np.zeros(C); pairs = []
        for k, b in enumerate(BANDS):
            v = imp[k * n_per:(k + 1) * n_per]
            for j, (a, c) in enumerate(zip(*iu)):
                per_ch[a] += v[j] / 2; per_ch[c] += v[j] / 2
                pairs.append((float(v[j]), b, ch[a], ch[c]))
        pairs.sort(reverse=True)
        diag = float(sum(imp[k * n_per:(k + 1) * n_per][iu[0] == iu[1]].sum() for k in range(4)))
        out[exp] = {'n_folds': len(splits), 'importance': 'LightGBM gain, normalised to sum 1, mean over folds',
                    'per_mode': per_mode,
                    'per_channel': dict(sorted({c: float(v) for c, v in zip(ch, per_ch)}.items(),
                                               key=lambda kv: -kv[1])),
                    'share_variance_terms': diag, 'share_covariance_terms': 1 - diag,
                    'top20_features': [{'gain_share': p[0], 'mode': p[1], 'ch1': p[2], 'ch2': p[3],
                                        'type': 'variance' if p[2] == p[3] else 'covariance'} for p in pairs[:20]]}
        logger.info(f'{exp}: per mode {json.dumps({k: round(v, 3) for k, v in per_mode.items()})}; '
                    f'top channels {list(out[exp]["per_channel"])[:5]}')
    save('importance_main_pipeline', out)


# ------------------------------------------------------------------ runtime
def part_runtime(cfg, seed, n_jobs):
    from pyriemann.estimation import Covariances
    from src.smvmd import SMVMD
    W, *_ = load_windows(cfg, 'DS2')
    X, y, g = load(cfg, 'DS2')
    rng = np.random.default_rng(seed); idx = rng.choice(len(W), 50, replace=False)
    sm = SMVMD(alpha=cfg['ds2']['smvmd']['alpha'], tau=0.0, tol=1e-10, max_iter=500)
    t0 = time.perf_counter()
    res = [sm.decompose(W[i].astype(np.float64)) for i in idx]
    t_smvmd = (time.perf_counter() - t0) / len(idx)
    comp = np.stack([np.stack([sum([m for m, f in zip(r.mimfs, r.center_freqs_hz) if lo <= f < hi],
                                   np.zeros(W.shape[1:])) for _, lo, hi in s2c.BANDS]) for r in res])
    t0 = time.perf_counter()
    for b in range(4):
        Covariances(estimator='oas').transform(comp[:, b])
    t_cov = (time.perf_counter() - t0) / len(idx)
    tr = np.flatnonzero(~np.isin(g, np.unique(g)[:12]))      # ~90 % of the children, as in one outer fold
    t0 = time.perf_counter(); m = fixed_pipe().fit(X[tr], y[tr]); t_train = time.perf_counter() - t0
    Xt = X[idx]
    t0 = time.perf_counter()
    for _ in range(5):
        m.predict_proba(Xt)
    t_pred = (time.perf_counter() - t0) / (5 * len(idx))
    bst = m.named_steps['clf'].model_.booster_
    trees = bst.dump_model()['tree_info']
    size = len(pickle.dumps(m))
    out = {'machine': 'laptop CPU, single thread per stage', 'n_windows_timed': len(idx),
           'per_window_ms': {'smvmd': 1000 * t_smvmd, 'covariances_4_modes': 1000 * t_cov,
                             'tangent_space_plus_lightgbm': 1000 * t_pred,
                             'total': 1000 * (t_smvmd + t_cov + t_pred)},
           'window_length_s': 5.0,
           'train_seconds_one_model': t_train, 'train_windows': int(len(tr)),
           'n_features': int(4 * X.shape[-1] * (X.shape[-1] + 1) // 2),
           'n_trees': len(trees), 'n_leaves_total': int(sum(t['num_leaves'] for t in trees)),
           'pickled_model_kb': size / 1024,
           'note': 'SMVMD dominates; it was parallelised over windows for the experiments.'}
    logger.info(json.dumps(out, indent=1))
    save('runtime_main_pipeline', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['all', 'perm', 'gap', 'importance', 'runtime'])
    ap.add_argument('--n-jobs', type=int, default=6)
    ap.add_argument('--n-perm', type=int, default=200)
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml'); seed = int(cfg.get('random_seed', 42)); set_random_seed(seed)
    parts = ['runtime', 'importance', 'gap', 'perm'] if a.part == 'all' else [a.part]
    for p in parts:
        t0 = time.perf_counter(); logger.info(f'===== part {p} =====')
        if p == 'perm':
            part_perm(cfg, seed, a.n_jobs, a.n_perm)
        else:
            globals()[f'part_{p}'](cfg, seed, a.n_jobs)
        logger.info(f'===== part {p} done in {time.perf_counter() - t0:.0f}s =====')


if __name__ == '__main__':
    main()
