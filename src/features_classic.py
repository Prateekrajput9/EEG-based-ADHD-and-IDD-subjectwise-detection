"""Standard EEG features computed per window (Stage 2 baselines).

'spectral' set, per channel (16 values):
    log10 absolute band power  delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
    relative band power        same 5 bands / total 1-45 Hz power
    log10 theta/beta ratio
    Hjorth activity (log10), mobility, complexity
    spectral entropy           normalised Shannon entropy of the 1-45 Hz PSD
    sample entropy             m = 2, r = 0.2 * SD of the window
'connectivity' set, per channel pair and band (theta, alpha, beta):
    magnitude-squared coherence (band mean)
    weighted phase-lag index (wPLI) over the window's samples (Hilbert phase)

PSD: Welch, Hann, 256-sample segments (2 s), 50% overlap -> 0.5 Hz bins.
No step uses information from other windows, so these features cannot leak
across children; scaling is done later inside each training fold.
Note: DS-1 cleaned data was band-passed 1-30 Hz by the dataset authors, so its
gamma band holds almost no signal.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, hilbert, sosfiltfilt, welch

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
CONN_BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
CACHE = Path('results/cache/features')
VERSION = 'v1'


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    x = np.asarray(x, dtype=np.float64)
    r = r_factor * x.std()
    n = len(x) - m
    if r == 0 or n < 2:
        return 0.0
    emb = np.lib.stride_tricks.sliding_window_view(x, m + 1)[:n]   # (n, m+1)
    d = np.zeros((n, n))
    for k in range(m):
        np.maximum(d, np.abs(emb[:, k, None] - emb[None, :, k]), out=d)
    B = (d <= r).sum() - n                        # template matches of length m (no self)
    np.maximum(d, np.abs(emb[:, m, None] - emb[None, :, m]), out=d)
    A = (d <= r).sum() - n                        # ... of length m + 1
    if A == 0 or B == 0:
        return float(np.log(n * (n - 1)))         # undefined; capped at the max attainable
    return float(-np.log(A / B))


def _hjorth(x):
    dx = np.diff(x, axis=-1); ddx = np.diff(dx, axis=-1)
    v, vd, vdd = x.var(-1), dx.var(-1), ddx.var(-1)
    mob = np.sqrt(vd / (v + 1e-20)); comp = np.sqrt(vdd / (vd + 1e-20)) / (mob + 1e-20)
    return np.log10(v + 1e-20), mob, comp


def spectral_one(w: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """w: (C, T) -> (C * 16,) channel-major."""
    f, p = welch(w, fs=fs, nperseg=256, noverlap=128, axis=-1)
    df = f[1] - f[0]
    tot_m = (f >= 1) & (f < 45)
    total = p[:, tot_m].sum(-1) * df + 1e-20
    absp = np.stack([p[:, (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in BANDS.values()], 1)
    rel = absp / total[:, None]
    tbr = np.log10((absp[:, 1] + 1e-20) / (absp[:, 3] + 1e-20))
    act, mob, comp = _hjorth(w)
    q = p[:, tot_m] / (p[:, tot_m].sum(-1, keepdims=True) + 1e-20)
    sent = -(q * np.log(q + 1e-20)).sum(-1) / np.log(q.shape[1])
    sampen = np.array([sample_entropy(ch) for ch in w])
    feats = np.column_stack([np.log10(absp + 1e-20), rel, tbr, act, mob, comp, sent, sampen])
    return feats.ravel()


def spectral_names(channels: List[str]) -> List[str]:
    per = ([f'logpow_{b}' for b in BANDS] + [f'relpow_{b}' for b in BANDS] +
           ['log_theta_beta', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
            'spectral_entropy', 'sample_entropy'])
    return [f'{c}|{n}' for c in channels for n in per]


_SOS = {b: butter(4, [lo, hi], btype='band', fs=128.0, output='sos') for b, (lo, hi) in CONN_BANDS.items()}


def connectivity_one(w: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """w: (C, T) -> (n_pairs * 6,): for each pair, coherence(theta, alpha, beta), wPLI(...)."""
    C, T = w.shape
    iu = np.triu_indices(C, 1)
    # coherence: Welch cross-spectra, 128-sample Hann segments, 50% overlap
    seg = np.lib.stride_tricks.sliding_window_view(w, 128, axis=-1)[:, ::64]   # (C, S, 128)
    X = np.fft.rfft(seg * np.hanning(128), axis=-1)                           # (C, S, F)
    f = np.fft.rfftfreq(128, 1 / fs)
    Sxx = (np.abs(X) ** 2).mean(1)                                             # (C, F)
    Sxy = np.einsum('isf,jsf->ijf', X, np.conj(X)) / X.shape[1]                # (C, C, F)
    msc = np.abs(Sxy) ** 2 / (Sxx[:, None, :] * Sxx[None, :, :] + 1e-30)
    coh = [msc[..., (f >= lo) & (f < hi)].mean(-1)[iu] for lo, hi in CONN_BANDS.values()]
    wpli = []
    for b in CONN_BANDS:
        z = hilbert(sosfiltfilt(_SOS[b], w, axis=-1), axis=-1)
        im = np.imag(z[iu[0]] * np.conj(z[iu[1]]))                            # (pairs, T)
        wpli.append(np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-30))
    return np.column_stack(coh + wpli).ravel()


def connectivity_names(channels: List[str]) -> List[str]:
    iu = np.triu_indices(len(channels), 1)
    per = [f'coh_{b}' for b in CONN_BANDS] + [f'wpli_{b}' for b in CONN_BANDS]
    return [f'{channels[i]}-{channels[j]}|{n}' for i, j in zip(*iu) for n in per]


# ── Stage 2b feature sets ────────────────────────────────────────────────────

FB = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
_SOS_FB = [butter(4, b, btype='band', fs=128.0, output='sos') for b in FB]


def fb_logvar_one(w: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """Filter-bank log-variance: 6 bands x C channels (channel-major)."""
    v = np.stack([np.log10(sosfiltfilt(s, w, axis=-1).var(-1) + 1e-20) for s in _SOS_FB], 1)
    return v.ravel()


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    emb = np.lib.stride_tricks.sliding_window_view(x, (order - 1) * delay + 1)[:, ::delay]
    pat = np.argsort(emb, axis=1, kind='stable')
    codes = (pat * (order ** np.arange(order))).sum(1)
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    from math import factorial, log
    return float(-(p * np.log(p)).sum() / log(factorial(order)))


def aperiodic_one(w: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """Per channel: aperiodic exponent and offset (log-log PSD fit, 2-25 Hz without
    7-14 Hz so the alpha peak does not bias it; 2-25 Hz lies inside both datasets'
    pass-bands), spectral entropy, sample entropy, permutation entropy (order 3, 5)."""
    f, p = welch(w, fs=fs, nperseg=256, noverlap=128, axis=-1)
    m = (f >= 2) & (f <= 25) & ~((f >= 7) & (f <= 14))
    lf, lp = np.log10(f[m]), np.log10(p[:, m] + 1e-20)
    A = np.column_stack([lf, np.ones_like(lf)])
    coef = np.linalg.lstsq(A, lp.T, rcond=None)[0]                  # (2, C)
    tot = (f >= 1) & (f < 45)
    q = p[:, tot] / (p[:, tot].sum(-1, keepdims=True) + 1e-20)
    sent = -(q * np.log(q + 1e-20)).sum(-1) / np.log(q.shape[1])
    se = [sample_entropy(ch) for ch in w]
    pe3 = [permutation_entropy(ch, 3) for ch in w]; pe5 = [permutation_entropy(ch, 5) for ch in w]
    return np.column_stack([-coef[0], coef[1], sent, se, pe3, pe5]).ravel()


def wpt_one(w: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """Wavelet packet (db4, level 4 -> 16 sub-bands of 4 Hz): log energy and relative
    energy per sub-band and channel."""
    import pywt
    rows = []
    for ch in w:
        wp = pywt.WaveletPacket(ch, 'db4', mode='symmetric', maxlevel=4)
        e = np.array([np.sum(n.data ** 2) for n in wp.get_level(4, order='freq')])
        rows.append(np.concatenate([np.log10(e + 1e-20), e / (e.sum() + 1e-20)]))
    return np.concatenate(rows)


FUNCS = {'spectral': spectral_one, 'connectivity': connectivity_one, 'fb_logvar': fb_logvar_one,
         'aperiodic': aperiodic_one, 'wpt': wpt_one}


def _batch(fn, Wb):
    return np.vstack([fn(w) for w in Wb])


def compute(W: np.ndarray, kind: str, window_key: str, exp: str, n_jobs: int = -1) -> np.ndarray:
    """Feature matrix for every window, cached by (exp, window cache key, kind, version)."""
    key = hashlib.sha1(f'{exp}|{window_key}|{kind}|{VERSION}'.encode()).hexdigest()[:12]
    path = CACHE / f'{exp}_{kind}_{key}.npy'
    if path.exists():
        return np.load(path)
    fn = FUNCS[kind]
    batches = np.array_split(np.arange(len(W)), max(1, len(W) // 32))
    X = np.vstack(Parallel(n_jobs=n_jobs)(delayed(_batch)(fn, W[b]) for b in batches))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X
