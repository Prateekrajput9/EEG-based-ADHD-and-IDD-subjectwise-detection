"""Stage 2c: SMVMD common modes -> Riemannian tangent space (supervisor's suggestion).

Idea: Step A's adaptive decomposition (SMVMD) combined with the best-performing
classifier family (Riemannian tangent space of channel covariances).

1. SMVMD is run on every window with the SAME settings as Step A (alpha per
   dataset, tau = 0, tol = 1e-10). It returns a window-specific number of
   multivariate modes, each with one centre frequency shared by all channels.
2. "Common modes": because the number and position of modes differ between
   windows, the modes are mapped to four fixed bands by their centre frequency
   (delta < 4 Hz, theta 4-8, alpha 8-13, beta 13-30 Hz) and summed within each
   band. Every window then has the same four components (C x T each). Modes at
   >= 30 Hz are dropped (DS-1 is band-limited to 1-30 Hz by the dataset authors;
   in DS-2 they carry < 1 % of the energy and are absent from most windows).
   A fifth component is the full SMVMD reconstruction (sum of all modes).
3. One OAS covariance per component and window. This is a per-window,
   label-free computation (no statistics across windows or people), so it is
   precomputed once without leakage. If a window has no mode in a band, the
   component is zero; every covariance gets a floor of EPS * (mean channel
   power of the whole window) * I so that it stays positive definite. The
   fraction of such empty bands is reported.
4. Tangent space (Riemannian mean fitted on the TRAINING windows of each fold),
   standardisation and LR / SVM, under the usual nested subject-wise protocol.

Variants (all inductive, nothing uses test-subject data):
  smvmdcm_<band>_ts_<clf>   one common mode
  smvmdcm_all_ts_<clf>      all four common modes, tangent vectors concatenated
  smvmd_recon_ts_<clf>      SMVMD reconstruction (all modes summed)
  smvmdcm_select_ts_<clf>   which mode(s) to use is chosen by the INNER CV on the
                            training subjects only (honest "best mode")
  fbbw_<band>_ts_<clf>, fbbw_all_ts_<clf>
                            control: identical pipeline on zero-phase Butterworth
                            bands with the same edges (is SMVMD needed at all?)

Per-mode rows are an analysis: picking the best of them by its test score would
be optimistic; the *_select_* rows are the leakage-free answer.
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # SMVMD workers are single-threaded; avoids oversubscription

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.protocol import runner
from src.smvmd import decompose_segments
from src.utils import load_config, set_random_seed, setup_logger
from src.windows import load_windows

OUT = Path('results/stage2c')
CACHE = Path('results/cache/smvmd_riemann')
EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
BANDS = [('delta', 0.0, 4.0), ('theta', 4.0, 8.0), ('alpha', 8.0, 13.0), ('beta', 13.0, 30.0)]
BW_EDGES = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0)}
EPS = 1e-4
CHUNK = 400
logger = setup_logger('Stage2c', 'results/logs/stage2c_smvmd_riemann.log')


# ---------------------------------------------------------------- covariances
def _oas(X):
    from pyriemann.estimation import Covariances
    return Covariances(estimator='oas').transform(X.astype(np.float64))


def _floor(C, power):
    """C: (n, C, C); power: (n,) mean channel power of the whole window."""
    eye = np.eye(C.shape[-1])[None]
    return C + EPS * power[:, None, None] * eye


def compute_covs(W, alpha, n_jobs):
    n, C, T = W.shape
    cs = np.zeros((n, len(BANDS) + 1, C, C))          # 4 common modes + reconstruction
    cb = np.zeros((n, len(BANDS), C, C))              # Butterworth control
    cf = np.full((n, 20), np.nan)
    nmodes = np.zeros(n, int)
    share = np.zeros((n, len(BANDS) + 1))             # energy share per band (+ >= 30 Hz)
    empty = np.zeros((n, len(BANDS)), bool)
    sos = {b: butter(4, BW_EDGES[b], btype='band', fs=128, output='sos') for b, *_ in BANDS}
    for s in range(0, n, CHUNK):
        t0 = time.perf_counter()
        Wc = W[s:s + CHUNK].astype(np.float64)
        Wc = Wc - Wc.mean(-1, keepdims=True)
        power = (Wc ** 2).mean((1, 2))
        res = decompose_segments(Wc, alpha=alpha, tau=0.0, tol=1e-10, max_iter=500,
                                 verbose=False, n_jobs=n_jobs)
        comp = np.zeros((len(Wc), len(BANDS) + 1, C, T))
        for i, r in enumerate(res):
            f = np.asarray(r.center_freqs_hz, float)
            M = np.stack(r.mimfs)                      # (K, C, T)
            e = (M ** 2).sum((1, 2))
            nmodes[s + i] = r.n_modes; cf[s + i, :len(f)] = f[:20]
            for b, (_, lo, hi) in enumerate(BANDS):
                sel = (f >= lo) & (f < hi)
                empty[s + i, b] = not sel.any()
                if sel.any():
                    comp[i, b] = M[sel].sum(0)
                share[s + i, b] = e[sel].sum() / e.sum()
            share[s + i, -1] = e[f >= 30].sum() / e.sum()
            comp[i, -1] = M.sum(0)
        for b in range(len(BANDS) + 1):
            cs[s:s + CHUNK, b] = _floor(_oas(comp[:, b]), power)
        for b, (name, *_) in enumerate(BANDS):
            cb[s:s + CHUNK, b] = _floor(_oas(sosfiltfilt(sos[name], Wc, axis=-1)), power)
        logger.info(f'  windows {s}-{s + len(Wc)} of {n}: {time.perf_counter() - t0:.0f}s')
    return dict(covs_smvmd=cs, covs_bw=cb, centre_hz=cf, n_modes=nmodes, share=share, empty=empty)


def cache_path(cfg, exp):
    _, _, _, m = load_windows(cfg, exp)
    return CACHE / f'{exp}_{m["key"]}_eps{EPS:g}.npz'


def part_covs(cfg, seed, n_jobs):
    CACHE.mkdir(parents=True, exist_ok=True)
    for exp in ['DS1_RestMusic', 'DS2']:
        p = cache_path(cfg, exp)
        if p.exists():
            logger.info(f'{exp}: covariance cache exists ({p})'); continue
        W, y, g, m = load_windows(cfg, exp)
        alpha = cfg['ds2' if exp == 'DS2' else 'ds1']['smvmd']['alpha']
        t0 = time.perf_counter(); logger.info(f'{exp}: SMVMD on {len(W)} windows (alpha={alpha})')
        d = compute_covs(W, alpha, n_jobs)
        np.savez(p, y=y, groups=g, condition=m['condition'], start=m['start'], **d)
        logger.info(f'{exp}: done in {time.perf_counter() - t0:.0f}s -> {p}')
    # DS-1 single conditions are row subsets of Rest+Music (checked, not assumed)
    rm = np.load(cache_path(cfg, 'DS1_RestMusic'))
    Wrm, *_ = load_windows(cfg, 'DS1_RestMusic')
    for exp, cond in [('DS1_Rest', 'Rest'), ('DS1_Music', 'Music')]:
        p = cache_path(cfg, exp)
        if p.exists():
            continue
        W, y, g, m = load_windows(cfg, exp)
        idx = np.flatnonzero(np.char.lower(rm['condition'].astype(str)) == cond.lower())
        if not (len(idx) == len(W) and np.array_equal(Wrm[idx], W) and (rm['y'][idx] == y).all()
                and (rm['groups'][idx].astype(str) == g).all()):
            raise RuntimeError(f'{exp}: windows are not the {cond} rows of DS1_RestMusic')
        np.savez(p, y=y, groups=g, condition=m['condition'], start=m['start'],
                 **{k: rm[k][idx] for k in ('covs_smvmd', 'covs_bw', 'centre_hz', 'n_modes', 'share', 'empty')})
        logger.info(f'{exp}: sliced {len(idx)} rows from DS1_RestMusic')


def part_summary(cfg, seed, n_jobs):
    """Descriptive statistics of the common modes (no labels used except group sizes)."""
    out = {}
    for exp in EXPS:
        d = np.load(cache_path(cfg, exp))
        cf = d['centre_hz'][~np.isnan(d['centre_hz'])]
        out[exp] = {
            'n_windows': int(len(d['y'])),
            'modes_per_window_mean': float(d['n_modes'].mean()),
            'modes_per_window_min': int(d['n_modes'].min()), 'modes_per_window_max': int(d['n_modes'].max()),
            'pct_windows_without_mode': {b: float(100 * d['empty'][:, i].mean()) for i, (b, *_) in enumerate(BANDS)},
            'energy_share_pct_mean': {**{b: float(100 * d['share'][:, i].mean()) for i, (b, *_) in enumerate(BANDS)},
                                      'above30': float(100 * d['share'][:, -1].mean())},
            'centre_hz_hist_edges': [0, 1, 4, 8, 13, 20, 30, 45, 64.1],
            'centre_hz_hist_counts': np.histogram(cf, [0, 1, 4, 8, 13, 20, 30, 45, 64.1])[0].tolist(),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'common_modes_summary.json').write_text(json.dumps(out, indent=2))
    logger.info(json.dumps(out, indent=2))


# ---------------------------------------------------------------- models
class BandTangentSpace(BaseEstimator, TransformerMixin):
    """X: (n, B, C, C). Fits one Riemannian tangent space per selected band on the
    training windows and concatenates the tangent vectors."""

    def __init__(self, bands=(0,)):
        self.bands = bands

    def fit(self, X, y=None):
        from pyriemann.tangentspace import TangentSpace
        self.ts_ = [TangentSpace(metric='riemann').fit(X[:, b]) for b in self.bands]
        return self

    def transform(self, X):
        return np.hstack([ts.transform(X[:, b]) for ts, b in zip(self.ts_, self.bands)])


class BalancedLGBM(BaseEstimator):
    """LightGBM with balanced class weights; same budget as BalancedXGB (300 trees, lr 0.05,
    row subsample 0.8, column subsample 0.5). Depth is tuned in the inner CV."""

    def __init__(self, max_depth=3, n_estimators=300, learning_rate=0.05, seed=42):
        self.max_depth = max_depth; self.n_estimators = n_estimators
        self.learning_rate = learning_rate; self.seed = seed

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        self.model_ = LGBMClassifier(max_depth=self.max_depth, num_leaves=2 ** self.max_depth - 1,
                                     n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                     class_weight='balanced', n_jobs=1, random_state=self.seed,
                                     verbose=-1)
        self.model_.fit(X, np.asarray(y).astype(int))
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


CLF_NAMES = {'lr': 'LR', 'svm': 'SVM', 'xgb': 'XGBoost', 'lgbm': 'LightGBM'}


def head(clf):
    if clf == 'lr':
        return LogisticRegression(class_weight='balanced', max_iter=3000), {'clf__C': [0.01, 0.1, 1.0]}
    if clf == 'svm':
        return SVC(kernel='rbf', class_weight='balanced'), {'clf__C': [0.1, 1.0, 10.0]}
    if clf == 'xgb':
        from src.models.classic import BalancedXGB          # same settings as the Stage 2 XGBoost rows
        return BalancedXGB(), {'clf__max_depth': [2, 4]}
    if clf == 'lgbm':
        return BalancedLGBM(), {'clf__max_depth': [2, 4]}
    raise ValueError(clf)


def pipe(bands, clf):
    h, grid = head(clf)
    return (Pipeline([('ts', BandTangentSpace(bands=tuple(bands))), ('scale', StandardScaler()), ('clf', h)]),
            dict(grid))


VARIANTS = ['bands', 'all', 'recon', 'select', 'fbbw_bands', 'fbbw_all', 'raw']


def raw_covs(cfg, exp):
    """Broadband OAS covariance of the window itself (the Step G input), shape (n, 1, C, C)."""
    W, *_ = load_windows(cfg, exp)
    return _oas(W)[:, None]


def part_eval(cfg, seed, n_jobs, exps=None, clfs=('lr', 'svm'), variants=None):
    names = [b for b, *_ in BANDS]
    variants = set(variants or [v for v in VARIANTS if v != 'raw'])   # LR/SVM raw rows exist in Stage 2b
    for exp in exps or EXPS:
        d = np.load(cache_path(cfg, exp))
        y, g = d['y'].astype(int), d['groups'].astype(str)
        CS, CB = d['covs_smvmd'], d['covs_bw']
        CR = raw_covs(cfg, exp) if 'raw' in variants else None
        data = {'cache': str(cache_path(cfg, exp)), 'eps_floor': EPS,
                'bands_hz': {b: [lo, hi] for b, lo, hi in BANDS}}
        jobs = []
        for clf in clfs:
            cname = CLF_NAMES[clf]
            mk = lambda f: {'name': f'Tangent space + {cname}', 'features': f}
            if 'bands' in variants:
                for i, b in enumerate(names):
                    jobs.append((f'smvmdcm_{b}_ts_{clf}', CS, (i,), None, mk(f'SMVMD common mode ({b})')))
            if 'all' in variants:
                jobs.append((f'smvmdcm_all_ts_{clf}', CS, (0, 1, 2, 3), None, mk('SMVMD common modes (all four)')))
            if 'recon' in variants:
                jobs.append((f'smvmd_recon_ts_{clf}', CS, (4,), None, mk('SMVMD reconstruction (all modes)')))
            if 'select' in variants:
                jobs.append((f'smvmdcm_select_ts_{clf}', CS, (0,), [(0,), (1,), (2,), (3,), (0, 1, 2, 3)],
                             mk('SMVMD common mode chosen by inner CV (training subjects only)')))
            if 'fbbw_bands' in variants:
                for i, b in enumerate(names):
                    jobs.append((f'fbbw_{b}_ts_{clf}', CB, (i,), None, mk(f'Butterworth band ({b}), control')))
            if 'fbbw_all' in variants:
                jobs.append((f'fbbw_all_ts_{clf}', CB, (0, 1, 2, 3), None,
                             mk('Butterworth bands (all four), control')))
            if 'raw' in variants:
                jobs.append((f'raw_ts_{clf}', CR, (0,), None,
                             mk('broadband covariance, no decomposition (Step G input)')))
        for name, X, bands, band_grid, desc in jobs:
            clf = name.rsplit('_', 1)[1]
            p, grid = pipe(bands, clf)
            if band_grid:
                grid = {'ts__bands': band_grid, **grid}
            runner.evaluate(OUT, exp, name, p, X, y, g, desc, seed=seed, grid=grid, n_jobs=n_jobs,
                            data_desc=data, logger=logger)


def part_table(cfg, seed, n_jobs):
    from src.protocol.tables import collect, write
    rows = collect([str(OUT)])
    ref = [r for r in collect(['results/stage2b'])
           if Path(r['source']).name.startswith(('riemann_ts_lr', 'riemann_ts_svm'))]
    for r in ref:
        r['features'] = 'broadband covariance (Step G reference)'
    from src.protocol.tables import ORDER
    rows = sorted(rows + ref, key=lambda r: (ORDER.index(r['experiment']), -(r['window_bal_acc'] or 0)))
    write(rows, OUT / 'comparison', 'Stage 2c: SMVMD common modes + Riemannian tangent space',
          note='Per-mode rows are an analysis; the "chosen by inner CV" rows pick the mode inside the '
               'training folds and are the leakage-free answer. Step G rows are included for reference.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['all', 'covs', 'summary', 'eval', 'table'])
    ap.add_argument('--exp', nargs='*', default=None)
    ap.add_argument('--clf', nargs='*', default=['lr', 'svm'], choices=list(CLF_NAMES))
    ap.add_argument('--variants', nargs='*', default=None, choices=VARIANTS)
    ap.add_argument('--n-jobs', type=int, default=8)
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml'); seed = int(cfg.get('random_seed', 42)); set_random_seed(seed)
    parts = ['covs', 'summary', 'eval', 'table'] if a.part == 'all' else [a.part]
    for p in parts:
        t0 = time.perf_counter(); logger.info(f'===== part {p} =====')
        if p == 'eval':
            part_eval(cfg, seed, a.n_jobs, a.exp, tuple(a.clf), a.variants)
        else:
            globals()[f'part_{p}'](cfg, seed, a.n_jobs)
        logger.info(f'===== part {p} done in {time.perf_counter() - t0:.0f}s =====')


if __name__ == '__main__':
    main()
