"""Stage 2b: accuracy-improvement sweep under the same nested subject-wise protocol.

Every method is reported, not only the best. Methods marked TRANSDUCTIVE use
unlabelled windows of the TEST child at test time (its own statistics); they are
flagged in every table with (†) and are not comparable to inductive rows. No
method sees test-child labels, and no hyper-parameter is tuned on test children.

  persubj_spectral_lr   (†) spectral features z-scored within each child
  riemann_ts_lr/svm         covariance (OAS) -> Riemannian tangent space -> LR / SVM
  ea_riemann_ts_lr      (†) Euclidean alignment per child, then tangent space -> LR
  fb_logvar_lr              filter-bank (6 bands) log-variance -> LR
  aperiodic_lr / _xgb       1/f exponent + offset, spectral/sample/permutation entropy
  wpt_lr                    wavelet-packet (db4, level 4) sub-band energies -> LR
  childagg_spectral_lr      one vector per child (mean + SD of spectral features
                            over its windows) -> LR; window metrics = child metrics
  raw_eegnet_ea         (†) EEGNet on Euclidean-aligned windows (augmentation on)
  minirocket_ridge          MiniRocket (10k kernels) -> ridge classifier
  ensemble_prob_avg         mean P(disorder) of a member list FIXED IN ADVANCE (below)

(EEGNet / ShallowConvNet with augmentation, without alignment, are Stage 2 rows.)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import features_classic as fc
from src.models import classic
from src.protocol import runner, summarize
from src.protocol.results import save_result
from src.utils import load_config, set_random_seed, setup_logger
from src.windows import load_windows

OUT = Path('results/stage2b')
EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
logger = setup_logger('Stage2b', 'results/logs/stage2b_sweep.log')

# Fixed before any Stage 2b result was seen (no selection on test performance).
ENSEMBLE_MEMBERS = [('results/stage1_eval', 'knn_tuned'), ('results/stage2', 'spectral_lr'),
                    ('results/stage2', 'spectral_xgb'), ('results/stage2', 'raw_eegnet'),
                    ('results/stage2b', 'riemann_ts_lr')]


def lr():
    return (Pipeline([('scale', StandardScaler()),
                      ('clf', LogisticRegression(class_weight='balanced', max_iter=3000))]),
            {'clf__C': [0.01, 0.1, 1.0]})


def per_child_zscore(X, g):
    """TRANSDUCTIVE: each child is standardised with its own windows' mean/SD."""
    Z = np.empty_like(X, dtype=np.float64)
    for s in np.unique(g):
        m = g == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)
    return Z


def euclidean_align(W, g):
    """TRANSDUCTIVE (He & Wu, IEEE TBME 2019): whiten each child's windows by the
    inverse square root of that child's mean spatial covariance."""
    # DS-1's cleaned recordings are rank-deficient (ICA components were removed by
    # the dataset authors), so R is shrunk towards (trace/C) I before inversion.
    A = np.empty(W.shape, dtype=np.float64)
    for s in np.unique(g):
        m = np.flatnonzero(g == s)
        Xc = W[m].astype(np.float64)
        Xc = Xc - Xc.mean(-1, keepdims=True)
        R = np.einsum('nct,ndt->cd', Xc, Xc) / (len(m) * Xc.shape[-1])
        R = 0.99 * R + 0.01 * np.trace(R) / len(R) * np.eye(len(R))
        ev, V = np.linalg.eigh(R)
        P = (V / np.sqrt(ev)) @ V.T                       # R^(-1/2), symmetric
        A[m] = np.einsum('cd,ndt->nct', P, W[m].astype(np.float64))
    return A


def riemann(clf='lr'):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    head = (LogisticRegression(class_weight='balanced', max_iter=3000) if clf == 'lr'
            else SVC(kernel='rbf', class_weight='balanced'))
    pipe = Pipeline([('cov', Covariances(estimator='oas')), ('ts', TangentSpace(metric='riemann')),
                     ('scale', StandardScaler()), ('clf', head)])
    return pipe, {'clf__C': [0.01, 0.1, 1.0] if clf == 'lr' else [0.1, 1.0, 10.0]}


class F64(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class MiniRocketFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=10000, random_state=42):
        self.num_kernels = num_kernels; self.random_state = random_state

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        self.mr_ = MiniRocketMultivariate(num_kernels=self.num_kernels, random_state=self.random_state)
        self.mr_.fit(np.asarray(X, dtype=np.float32))
        return self

    def transform(self, X):
        return np.asarray(self.mr_.transform(np.asarray(X, dtype=np.float32)), dtype=np.float32)


def feats(cfg, exp, kind):
    W, y, g, m = load_windows(cfg, exp)
    return fc.compute(W, kind, m['key'], exp), y, g, m


def ev(exp, name, model, X, y, g, desc, seed, grid=None, **kw):
    return runner.evaluate(OUT, exp, name, model, X, y, g, desc, seed=seed, grid=grid,
                           logger=logger, **kw)


def part_features(cfg, seed):
    for exp in EXPS:
        for kind in ('fb_logvar', 'aperiodic', 'wpt'):
            t0 = time.perf_counter(); X, *_ = feats(cfg, exp, kind)
            logger.info(f'{exp} {kind}: {X.shape} [{time.perf_counter() - t0:.0f}s]')


def part_classic(cfg, seed):
    for exp in EXPS:
        W, y, g, m = load_windows(cfg, exp)
        Xs = fc.compute(W, 'spectral', m['key'], exp)
        p, grid = lr()
        ev(exp, 'persubj_spectral_lr', p, per_child_zscore(Xs, g), y, g,
           {'name': 'Logistic regression', 'features': 'spectral, z-scored within each child',
            'uses_unlabelled_test_child_data': True}, seed, grid)
        for clf in ('lr', 'svm'):
            p, grid = riemann(clf)
            ev(exp, f'riemann_ts_{clf}', p, W.astype(np.float64), y, g,
               {'name': f'Tangent space + {"LR" if clf == "lr" else "SVM"}',
                'features': 'Riemannian tangent space (OAS covariance)'}, seed, grid)
        p, grid = riemann('lr')
        ev(exp, 'ea_riemann_ts_lr', p, euclidean_align(W, g).astype(np.float64), y, g,
           {'name': 'Tangent space + LR', 'features': 'Euclidean-aligned covariance, tangent space',
            'uses_unlabelled_test_child_data': True}, seed, grid)
        for kind in ('fb_logvar', 'aperiodic', 'wpt'):
            X = fc.compute(W, kind, m['key'], exp)
            p, grid = lr()
            ev(exp, f'{kind}_lr', p, X, y, g, {'name': 'Logistic regression', 'features': kind}, seed, grid)
        X = fc.compute(W, 'aperiodic', m['key'], exp)
        p, grid = classic.make('xgb', seed)
        ev(exp, 'aperiodic_xgb', p, X, y, g, {'name': 'XGBoost', 'features': 'aperiodic'}, seed, grid)
        # child-level aggregation: one row per child
        subs = np.array(sorted(set(g)))
        Xc = np.vstack([np.concatenate([Xs[g == s].mean(0), Xs[g == s].std(0)]) for s in subs])
        yc = np.array([int(y[g == s][0]) for s in subs])
        p, grid = lr()
        ev(exp, 'childagg_spectral_lr', p, Xc, yc, subs,
           {'name': 'Logistic regression (child level)', 'features': 'spectral mean+SD per child',
            'unit': 'child (window metrics equal child metrics)'}, seed, grid,
           inner_n_splits=3 if exp.startswith('DS1') else 4)


def part_deep(cfg, seed):
    from src.models.deep import TorchWindowClassifier
    for exp in EXPS:
        W, y, g, m = load_windows(cfg, exp)
        kw = {'max_epochs': 30, 'patience': 6} if exp == 'DS2' else {}
        model = TorchWindowClassifier(arch='eegnet', seed=seed, n_threads=2, **kw)
        ev(exp, 'raw_eegnet_ea', model, euclidean_align(W, g), y, g,
           {'name': 'EEGNet-8,2', 'features': 'Euclidean-aligned raw windows', 'params': model.get_params(),
            'uses_unlabelled_test_child_data': True}, seed, pass_groups=True, n_jobs=4)


def part_minirocket(cfg, seed):
    for exp in EXPS:
        W, y, g, m = load_windows(cfg, exp)
        pipe = Pipeline([('mr', MiniRocketFeatures(random_state=seed)), ('scale', StandardScaler()),
                         ('clf', RidgeClassifierCV(alphas=np.logspace(-1, 4, 11), class_weight='balanced'))])
        ev(exp, 'minirocket_ridge', pipe, W, y, g,
           {'name': 'MiniRocket + ridge', 'features': 'MiniRocket 10k kernels (sktime)',
            'tuning': 'ridge alpha by efficient leave-one-window-out CV within the training children '
                      '(training children only; not subject-wise)'}, seed, n_jobs=1)


def part_ensemble(cfg, seed):
    from src.protocol.runner import scheme_for
    for exp in EXPS:
        sch = scheme_for(exp)
        paths = [Path(r) / exp / f'{n}__subject_{sch}.npz' for r, n in ENSEMBLE_MEMBERS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            logger.warning(f'{exp}: ensemble skipped, missing {missing}'); continue
        ds = [np.load(p) for p in paths]
        y, g, fold = ds[0]['y'], ds[0]['groups'].astype(str), ds[0]['fold']
        for d in ds[1:]:
            assert (d['y'] == y).all() and (d['groups'].astype(str) == g).all() and (d['fold'] == fold).all(), \
                'members must share the same windows and outer folds'
        score = np.mean([d['score'] for d in ds], axis=0)
        y_pred = (score >= 0.5).astype(int)
        s = summarize(y, y_pred, score, g, n_boot=runner.N_BOOT, seed=seed)
        desc = {'name': 'Ensemble (mean probability)', 'features': 'members fixed in advance',
                'members': [f'{r}/{n}' for r, n in ENSEMBLE_MEMBERS]}
        protocol = {'name': f'subject_{sch}', 'split': sch, 'leaky': False, 'seed': seed,
                    'nested': 'members individually nested or untuned'}
        save_result(OUT / exp, f'ensemble_prob_avg__subject_{sch}',
                    {'experiment': exp, 'model': desc, 'protocol': protocol, 'n_folds': int(fold.max() + 1),
                     'runtime_sec': 0.0, 'metrics': {'window': s['window'], 'child': s['child']},
                     'ci95': s['ci95'], 'per_child': s['per_child']},
                    {'y': y, 'y_pred': y_pred, 'score': score, 'fold': fold, 'groups': g},
                    {'dataset': 'DS2' if exp == 'DS2' else 'DS1', 'experiment': exp, 'model': desc,
                     'protocol': protocol}, seed)
        logger.info(f'[{exp} | ensemble] window {s["window"]["accuracy"]*100:.2f}% '
                    f'children {s["child"]["n_correct"]}/{s["child"]["n"]}')


def part_table(cfg, seed):
    from src.protocol.tables import collect, write
    rows = collect(['results/stage1_eval', 'results/stage2', str(OUT)])
    write(rows, OUT / 'comparison', 'Stage 2b: accuracy-improvement sweep (all methods)',
          note='Includes Stage 1 (SMVMD) and Stage 2 rows for reference. (†) = transductive.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all',
                    choices=['all', 'features', 'classic', 'deep', 'minirocket', 'ensemble', 'table'])
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml'); seed = int(cfg.get('random_seed', 42)); set_random_seed(seed)
    parts = (['features', 'classic', 'minirocket', 'deep', 'ensemble', 'table'] if a.part == 'all' else [a.part])
    for p in parts:
        t0 = time.perf_counter(); logger.info(f'===== part {p} =====')
        globals()[f'part_{p}'](cfg, seed)
        logger.info(f'===== part {p} done in {time.perf_counter() - t0:.0f}s =====')


if __name__ == '__main__':
    main()
