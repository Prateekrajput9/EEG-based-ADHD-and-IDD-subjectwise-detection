"""Standard result files.

Every evaluated (model, dataset, protocol) writes two files:
    <dir>/<name>.json  metrics, CIs, per-child table, folds, chosen params,
                       config + config hash, seed, git commit, runtime
    <dir>/<name>.npz   out-of-fold y, y_pred, score, fold, groups (for ensembles,
                       re-scoring and figures without refitting)
Tables and figures are built only from these files.
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

SCHEMA_VERSION = 1
DS2_CONFOUND = ('DS-2 confound: the ADHD children were taking methylphenidate and the '
                'controls were not, so ADHD cannot be separated from its treatment.')


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if callable(o):
        return getattr(o, '__name__', repr(o))
    if hasattr(o, 'get_params') and not isinstance(o, type):
        return type(o).__name__
    return o


def config_hash(cfg: Dict) -> str:
    return hashlib.sha1(json.dumps(jsonable(cfg), sort_keys=True).encode()).hexdigest()[:12]


def git_info() -> Dict[str, Any]:
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'experiments'],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return {'commit': commit, 'src_or_experiments_dirty': dirty}
    except Exception:
        return {'commit': None, 'src_or_experiments_dirty': None}


def save_result(out_dir, name: str, payload: Dict[str, Any], arrays: Dict[str, np.ndarray],
                config: Dict[str, Any], seed: int) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    doc = {'schema_version': SCHEMA_VERSION, 'name': name,
           'created_at': datetime.now().isoformat(timespec='seconds'),
           'seed': seed, 'config': config, 'config_hash': config_hash(config),
           'git': git_info(),
           'environment': {'python': sys.version.split()[0], 'numpy': np.__version__,
                           'platform': platform.platform()},
           **payload}
    if str(config.get('dataset', '')).upper().startswith('DS2'):
        doc.setdefault('notes', []).append(DS2_CONFOUND)
    path = out_dir / f'{name}.json'
    path.write_text(json.dumps(jsonable(doc), indent=1), encoding='utf-8')
    np.savez_compressed(out_dir / f'{name}.npz',
                        **{k: (np.asarray(v).astype(str) if k == 'groups' else np.asarray(v))
                           for k, v in arrays.items()})
    return path


def load_result(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))
