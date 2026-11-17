"""Windowed inference with a saved model bundle.

Bundle (joblib dict, written by experiments/train_demo_model.py):
    pipeline      fitted sklearn pipeline (features -> P(IDD))
    feature_kind  'spectral' (src/features_classic.py), computed per window exactly as offline
    channels, fs, window_sec
    honest        subject-wise (LOSO) metrics of the same recipe, read from its result JSON
"""
import time
from collections import deque

import joblib
import numpy as np
from scipy.signal import welch

from src import features_classic as fc

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}


class LiveModel:
    def __init__(self, path, smoothing_windows: int = 4):
        self.bundle = joblib.load(path)
        self.pipe = self.bundle['pipeline']
        self.kind = self.bundle['feature_kind']
        self.history = deque(maxlen=smoothing_windows)

    def features(self, window: np.ndarray) -> np.ndarray:
        return fc.FUNCS[self.kind](window.astype(np.float32))[None, :]

    def predict(self, window: np.ndarray, t_window_end: float = None) -> dict:
        t0 = time.perf_counter()
        x = np.nan_to_num(self.features(window), nan=0.0, posinf=0.0, neginf=0.0)
        p = float(self.pipe.predict_proba(x)[0, list(self.pipe.classes_).index(1)])
        self.history.append(p)
        t1 = time.perf_counter()
        return {'p_idd': p, 'p_idd_smoothed': float(np.mean(self.history)),
                'compute_ms': (t1 - t0) * 1000,
                'latency_ms': None if t_window_end is None else (t1 - t_window_end) * 1000}


def band_powers(window: np.ndarray, fs: float = 128.0) -> dict:
    """Relative band power averaged over channels (display only)."""
    f, p = welch(window, fs=fs, nperseg=256, noverlap=128, axis=-1)
    tot = p[:, (f >= 1) & (f < 45)].sum(-1) + 1e-20
    return {b: float(np.mean(p[:, (f >= lo) & (f < hi)].sum(-1) / tot)) for b, (lo, hi) in BANDS.items()}
