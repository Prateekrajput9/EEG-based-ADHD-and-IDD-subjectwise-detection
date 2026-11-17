"""Preprocessed EEG windows, cached to disk and aligned with the SMVMD feature cache.

Later stages (band powers, connectivity, EEGNet, Riemannian features, live
pipeline) start from the same preprocessed 5 s windows the SMVMD pipeline used:
    DS-1: the dataset's cleaned recordings (mode B), no further filtering
    DS-2: Butterworth 0.5-60 Hz, 50 Hz notch, Coiflet-3 + SURE (as src/preprocessing.py)
Row order matches results/features/<exp>_features.npz exactly (checked on load),
so per-window results from different feature sets can be combined.

Cache: results/cache/windows/<exp>_<hash>.npz, hash over every parameter that
changes the windows. DS-2 preprocessing takes ~12 min once; later loads ~1 s.
"""
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

CACHE = Path('results/cache/windows')
EXP_CONDITIONS = {'DS1_Rest': ['rest'], 'DS1_Music': ['music'],
                  'DS1_RestMusic': ['music', 'rest'], 'DS2': ['visual_task']}


def window_key(cfg: Dict, exp: str) -> str:
    seg = cfg['segmentation']
    if exp.startswith('DS1'):
        pre = f"mode={cfg['ds1']['preprocessing']['mode']}"
        fs = cfg['ds1']['sampling_freq']
    else:
        p = cfg['ds2']['preprocessing']
        pre = '|'.join(f'{k}={p[k]}' for k in sorted(p))
        fs = cfg['ds2']['sampling_freq']
    s = f"{exp}|fs={fs}|win={seg['window_sec']}|ovl={seg['overlap']}|{pre}|v1"
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _build_ds1(cfg, exp):
    from src.data_loader import DS1Loader
    from src.preprocessing import preprocess_ds1
    from src.segmentation import segment_eeg
    ds1 = cfg['ds1']; fs = ds1['sampling_freq']; mode = ds1['preprocessing']['mode']
    loader = DS1Loader(cfg['paths']['ds1_data_root'], mode=mode)
    files = loader.discover_files()
    W, Y, G, C, P = [], [], [], [], []
    for sid in sorted(files):                       # same order as src/pipeline.py
        for cond in sorted(c for c in files[sid] if c in EXP_CONDITIONS[exp]):
            eeg, info = loader.load_subject(sid, cond, files[sid][cond])
            eeg = preprocess_ds1(eeg, fs=fs, mode=mode, config=ds1['preprocessing'])
            segs, meta = segment_eeg(eeg=eeg, fs=fs, window_sec=cfg['segmentation']['window_sec'],
                                     overlap=cfg['segmentation']['overlap'], subject_id=sid,
                                     label=info['label'], dataset='DS1', condition=cond)
            W.append(segs); Y += [info['label']] * len(segs); G += [sid] * len(segs)
            C += [cond] * len(segs); P += [m.get('start_sample', i * 320) for i, m in enumerate(meta)]
    return np.concatenate(W).astype(np.float32), np.array(Y), np.array(G), np.array(C), np.array(P)


def _build_ds2(cfg):
    from src.data_loader import DS2Loader
    from src.preprocessing import preprocess_ds2
    from src.segmentation import segment_dataset_ds2
    ds2 = cfg['ds2']; fs = ds2['sampling_freq']
    eeg, meta = DS2Loader(cfg['paths']['ds2_adhd_dir']).load_all_subjects()
    pre = {s: preprocess_ds2(eeg=e, fs=fs, **dict(ds2['preprocessing'])) for s, e in eeg.items()}
    segs, df = segment_dataset_ds2(eeg_data=pre, metadata_list=meta, fs=fs,
                                   window_sec=cfg['segmentation']['window_sec'],
                                   overlap=cfg['segmentation']['overlap'])
    return (segs.astype(np.float32), df['label'].values.astype(int),
            np.asarray(df['subject_id'].astype(str).tolist(), dtype=str),
            np.array(['visual_task'] * len(df)), df['start_sample'].values.astype(int))


def load_windows(cfg: Dict, exp: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """(W, y, groups, meta): W is (n_windows, n_channels, 640) float32."""
    path = CACHE / f'{exp}_{window_key(cfg, exp)}.npz'
    if path.exists():
        d = np.load(path)
        W, y, g, cond, pos = d['W'], d['y'], d['groups'].astype(str), d['condition'].astype(str), d['start']
    else:
        W, y, g, cond, pos = _build_ds2(cfg) if exp == 'DS2' else _build_ds1(cfg, exp)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=W, y=y, groups=np.asarray(g, dtype=str), condition=np.asarray(cond, dtype=str), start=pos)
    ref = np.load(f'results/features/{exp}_features.npz')
    if not (len(y) == len(ref['y']) and (ref['y'] == y).all()
            and (ref['groups'].astype(str) == g).all()):
        raise RuntimeError(f'{exp}: windows are not aligned with the SMVMD feature cache')
    return W, y.astype(int), g, {'condition': cond, 'start': pos, 'cache': str(path),
                                 'fs': 128, 'key': window_key(cfg, exp)}
