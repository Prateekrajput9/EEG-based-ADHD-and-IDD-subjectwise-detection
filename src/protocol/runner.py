"""One call = evaluate a model under the standard protocol and save the result files."""
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .cv import cross_validate, nested_cv
from .metrics import summarize
from .results import save_result
from .splits import subject_splits

N_BOOT = 2000
log = logging.getLogger('protocol')


def scheme_for(exp: str) -> str:
    """LOSO for DS-1 (14 children), seeded StratifiedGroupKFold(10) for DS-2."""
    return 'loso' if exp.startswith('DS1') else 'sgkf'


def outer_splits(exp, y, g, seed):
    return subject_splits(y, g, scheme_for(exp), 10, seed)


def exists(out_root, exp, name) -> bool:
    return (Path(out_root) / exp / f'{name}__subject_{scheme_for(exp)}.json').exists()


def evaluate(out_root, exp: str, name: str, model, X, y, g, model_desc: Dict, seed: int = 42,
             grid: Optional[Dict] = None, inner_n_splits: int = 4, pass_groups: bool = False,
             n_jobs: int = -1, data_desc: Optional[Dict] = None, skip_existing: bool = True,
             logger=None):
    """Nested CV if `grid` is given, else plain subject-wise CV. Returns the summary."""
    logger = logger or log
    sch = scheme_for(exp)
    if skip_existing and exists(out_root, exp, name):
        logger.info(f'[{exp} | {name}] exists, skipped')
        return None
    splits = outer_splits(exp, y, g, seed)
    t0 = time.perf_counter()
    extra = {}
    if grid:
        res = nested_cv(model, grid, X, y, g, splits, inner_scheme='sgkf',
                        inner_n_splits=inner_n_splits, seed=seed, n_jobs=n_jobs,
                        pass_groups=pass_groups)
        extra = {'folds': res['folds'], 'grid': grid, 'grid_size': res['grid_size'],
                 'inner': res['inner'], 'criterion': res['criterion']}
    else:
        res = cross_validate(model, X, y, g, splits, n_jobs=n_jobs, pass_groups=pass_groups)
    summ = summarize(y, res['y_pred'], res['score'], g, n_boot=N_BOOT, seed=seed)
    protocol = {'name': f'subject_{sch}', 'split': sch, 'n_splits': 10 if sch != 'loso' else 'n_subjects',
                'leaky': False, 'seed': seed, 'nested': bool(grid)}
    payload = {'experiment': exp, 'model': model_desc, 'protocol': protocol,
               'n_folds': res['n_folds'], 'runtime_sec': round(time.perf_counter() - t0, 1),
               'metrics': {'window': summ['window'], 'child': summ['child']},
               'ci95': summ['ci95'], 'per_child': summ['per_child'], **extra}
    config = {'dataset': 'DS2' if exp == 'DS2' else 'DS1', 'experiment': exp,
              'data': data_desc or {}, 'model': model_desc, 'protocol': protocol, 'n_boot': N_BOOT,
              'grid': grid}
    path = save_result(Path(out_root) / exp, f'{name}__{protocol["name"]}', payload,
                       {'y': y, 'y_pred': res['y_pred'], 'score': res['score'],
                        'fold': res['fold'], 'groups': g}, config, seed)
    w, c = summ['window'], summ['child']
    logger.info(f"[{exp} | {name}] window acc {w['accuracy']*100:.2f}% bal {w['balanced_accuracy']*100:.2f}% "
                f"AUC {w['roc_auc']:.3f} | child {c['n_correct']}/{c['n']} | "
                f"{payload['runtime_sec']}s -> {path}")
    return summ
