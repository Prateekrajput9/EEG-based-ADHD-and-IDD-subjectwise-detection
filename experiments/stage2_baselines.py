"""Stage 2: standard baselines under the same nested subject-wise protocol.

Parts (python -m experiments.stage2_baselines --part <name>):
  windows   build/check the preprocessed window caches (DS-2 takes ~12 min once)
  features  spectral (band power, theta/beta, Hjorth, entropies) and connectivity
            (coherence, wPLI) features for every window, cached
  classic   {spectral, connectivity} x {LR, SVM, RF, XGBoost}, nested CV
  deep      EEGNet and ShallowConvNet on raw windows, subject-wise early stopping
  table     comparison table incl. the Stage 1 SMVMD + KNN rows -> results/stage2/

Protocol (src/protocol): outer LOSO (DS-1) / seeded StratifiedGroupKFold(10)
(DS-2); inner StratifiedGroupKFold(4) on outer-training children for tuning.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src import features_classic as fc
from src.models import classic
from src.protocol import runner
from src.utils import load_config, set_random_seed, setup_logger
from src.windows import load_windows

OUT = Path('results/stage2')
EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
logger = setup_logger('Stage2', 'results/logs/stage2_baselines.log')


def channels(cfg, exp):
    return cfg['ds1']['channel_names'] if exp.startswith('DS1') else cfg['ds2']['channel_names']


def part_windows(cfg, seed):
    for exp in EXPS:
        t0 = time.perf_counter()
        W, y, g, m = load_windows(cfg, exp)
        logger.info(f'{exp}: windows {W.shape}, {len(set(g))} children, aligned with SMVMD cache '
                    f'[{time.perf_counter() - t0:.0f}s] ({m["cache"]})')


def features(cfg, exp, kind):
    W, y, g, m = load_windows(cfg, exp)
    t0 = time.perf_counter()
    X = fc.compute(W, kind, m['key'], exp)
    return X, y, g, m, time.perf_counter() - t0


def part_features(cfg, seed):
    for exp in EXPS:
        for kind in ('spectral', 'connectivity'):
            X, y, g, m, dt = features(cfg, exp, kind)
            logger.info(f'{exp} {kind}: {X.shape} [{dt:.0f}s]')


def part_classic(cfg, seed):
    for exp in EXPS:
        for kind in ('spectral', 'connectivity'):
            X, y, g, m, _ = features(cfg, exp, kind)
            for clf in ('lr', 'svm', 'rf', 'xgb'):
                pipe, grid = classic.make(clf, seed)
                desc = {'name': classic.NAMES[clf], 'features': kind, 'n_features': X.shape[1],
                        'feature_definition': 'src/features_classic.py ' + fc.VERSION}
                runner.evaluate(OUT, exp, f'{kind}_{clf}', pipe, X, y, g, desc, seed=seed,
                                grid=grid, data_desc={'windows': m['key']}, logger=logger)


def part_deep(cfg, seed):
    from src.models.deep import TorchWindowClassifier
    for exp in EXPS:
        W, y, g, m = load_windows(cfg, exp)
        for arch, label in (('eegnet', 'EEGNet-8,2'), ('shallow', 'ShallowConvNet')):
            # DS-2 has ~10x more windows; cap epochs so CPU training stays tractable
            kw = {'max_epochs': 30, 'patience': 6} if exp == 'DS2' else {}
            model = TorchWindowClassifier(arch=arch, seed=seed, n_threads=2, **kw)
            desc = {'name': label, 'features': 'raw windows', 'params': model.get_params(),
                    'early_stopping': 'subject-wise 20% of training children, balanced accuracy'}
            runner.evaluate(OUT, exp, f'raw_{arch}', model, W, y, g, desc, seed=seed,
                            pass_groups=True, n_jobs=4, data_desc={'windows': m['key']}, logger=logger)


def part_table(cfg, seed):
    from src.protocol.tables import collect, write
    rows = collect(['results/stage1_eval', str(OUT)])
    write(rows, OUT / 'comparison', 'Stage 2: model comparison, subject-wise evaluation',
          note='SMVMD rows come from results/stage1_eval (Stage 1). All other rows: results/stage2.')
    logger.info(f'table: {len(rows)} rows -> {OUT / "comparison.md"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all',
                    choices=['all', 'windows', 'features', 'classic', 'deep', 'table'])
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml')
    seed = int(cfg.get('random_seed', 42)); set_random_seed(seed)
    parts = ['windows', 'features', 'classic', 'deep', 'table'] if a.part == 'all' else [a.part]
    for p in parts:
        t0 = time.perf_counter(); logger.info(f'===== part {p} =====')
        globals()[f'part_{p}'](cfg, seed)
        logger.info(f'===== part {p} done in {time.perf_counter() - t0:.0f}s =====')


if __name__ == '__main__':
    main()
