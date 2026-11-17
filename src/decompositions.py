"""Alternative decompositions for the Stage 3 ablation ladder.

Every rung produces, per window, a list of modes (each (C, T)); the SAME nine
features (src/features.py) are computed per mode and channel and combined with
the SAME energy-weighted integration (paper Eq. 7), so only the decomposition
changes:

  raw     one "mode" = the signal itself (no decomposition)
  bands   fixed Butterworth bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30,
          gamma 30-45 Hz (zero-phase, order 4)
  vmd     univariate VMD per channel, K = 5 modes (centre frequencies differ
          per channel)
  mvmd    multivariate VMD, K = 5 modes with one centre frequency per mode
          shared by all channels
  smvmd   the existing successive MVMD (adaptive K) - read from its cache

VMD/MVMD follow the same conventions as src/smvmd.py so the rungs are
comparable: same alpha per dataset, tau = 0 (no dual ascent), one-sided
spectrum, no mirror extension, uniform centre-frequency initialisation, at
most 500 ADMM iterations, stop when the relative mode change < 1e-7.
K = 5 is fixed (the paper's example segments show 4-5 modes; DS-2's adaptive
SMVMD averages 4.6).
"""
from typing import List

import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def bands(w: np.ndarray, fs: float = 128.0) -> List[np.ndarray]:
    return [sosfiltfilt(butter(4, b, btype='band', fs=fs, output='sos'), w, axis=-1) for b in BANDS]


def _vmd_core(x_hat: np.ndarray, K: int, alpha: float, shared: bool,
              max_iter: int = 500, tol: float = 1e-7) -> np.ndarray:
    """x_hat: (C, F) one-sided spectrum. Returns u_hat (K, C, F).

    shared=True  -> MVMD: omega_k common to all channels
    shared=False -> VMD run independently per channel (omega (K, C))
    """
    C, F = x_hat.shape
    freqs = np.linspace(0, 0.5, F)
    init = (np.arange(K) + 1) / (K + 1) * 0.5
    omega = init[:, None] if shared else np.repeat(init[:, None], C, axis=1)   # (K,1) or (K,C)
    u = np.zeros((K, C, F), dtype=complex)
    for _ in range(max_iter):
        u_old = u.copy()
        for k in range(K):
            others = u.sum(0) - u[k]
            u[k] = (x_hat - others) / (1.0 + 2.0 * alpha * (freqs[None, :] - omega[k][:, None]) ** 2)
            p = np.abs(u[k]) ** 2                                              # (C, F)
            if shared:
                omega[k] = (p * freqs).sum() / (p.sum() + 1e-20)
            else:
                omega[k] = (p * freqs).sum(1) / (p.sum(1) + 1e-20)
        change = np.sum(np.abs(u - u_old) ** 2) / (np.sum(np.abs(u_old) ** 2) + 1e-20)
        if change < tol:
            break
    return u


def _to_time(u_hat: np.ndarray, T: int) -> List[np.ndarray]:
    return [np.fft.irfft(m, n=T, axis=-1) for m in u_hat]


def vmd(w: np.ndarray, alpha: float, K: int = 5) -> List[np.ndarray]:
    return _to_time(_vmd_core(np.fft.rfft(w, axis=-1), K, alpha, shared=False), w.shape[-1])


def mvmd(w: np.ndarray, alpha: float, K: int = 5) -> List[np.ndarray]:
    return _to_time(_vmd_core(np.fft.rfft(w, axis=-1), K, alpha, shared=True), w.shape[-1])


def fast_gaussian_pair_sum(x: np.ndarray, sigma: float, block: int = 128) -> float:
    """Same value as src.features.gaussian_pair_sum (to ~1e-15 relative), about 2-3x
    faster: uses the symmetry of the kernel and in-place operations."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    c = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    s = 1.0 / (np.sqrt(2.0) * sigma)
    xs = x * s
    upper = 0.0
    for a in range(0, n, block):
        e = min(a + block, n)
        d = xs[a:e, None] - xs[None, a:]
        np.square(d, out=d); np.negative(d, out=d); np.exp(d, out=d)
        sq = d[:, :e - a]
        upper += d[:, e - a:].sum() + (sq.sum() - (e - a)) / 2.0     # strict upper of square part
    return float(c * (2.0 * upper + n))


def mode_features(modes: List[np.ndarray], fs: float = 128.0) -> np.ndarray:
    """9 features per (mode, channel) with the fast pair sum -> (C, K, 9)."""
    from src import features as F
    C = modes[0].shape[0]
    out = np.zeros((C, len(modes), 9))
    for k, m in enumerate(modes):
        for ch in range(C):
            x = m[ch].astype(np.float64)
            f = np.zeros(9)
            f[0] = F.compute_std(x); f[1] = F.compute_var(x); f[2] = F.compute_rms(x)
            f[3] = F.compute_iqr(x); f[4] = F.compute_assr(x); f[5] = F.compute_ap(x, fs=fs)
            f[6] = F.compute_mfl(x)
            sig = F.silverman_sigma(x)
            ps = fast_gaussian_pair_sum(x, sig) if sig is not None else None
            f[7] = F.compute_ip(x, sigma=sig, pair_sum=ps)
            f[8] = F.compute_pcc(x, sigma=sig, time_delay=1, pair_sum=ps)
            out[ch, k] = f
    return out


def integrate(modes: List[np.ndarray], feats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Paper Eq. 7: per channel, energy-weighted mean of each feature over modes."""
    e = np.stack([np.sum(m ** 2, axis=-1) for m in modes], axis=1)        # (C, K)
    w = e / (e.sum(1, keepdims=True) + eps)
    return (feats * w[:, :, None]).sum(1).ravel()                          # channel-major (C*9,)


def window_vector(w: np.ndarray, rung: str, alpha: float, fs: float = 128.0) -> np.ndarray:
    w = w.astype(np.float64)
    if rung == 'raw':
        modes = [w]
    elif rung == 'bands':
        modes = bands(w, fs)
    elif rung == 'vmd':
        modes = vmd(w, alpha)
    elif rung == 'mvmd':
        modes = mvmd(w, alpha)
    else:
        raise ValueError(rung)
    return integrate(modes, mode_features(modes, fs))
