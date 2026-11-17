"""Streaming preprocessor: the same windowing as offline training.

Offline (DS-1, mode B): the dataset's cleaned recording is cut into 5 s windows
with a 2.5 s step, starting at sample 0, no extra filtering
(src/segmentation.py). The replay path therefore applies NO filter, so a
replayed recording yields exactly the offline windows (tests/test_live_equivalence.py).

Headset path (raw EEG): optional causal DC removal, 1-30 Hz band-pass and 50 Hz
notch (state carried across chunks, so it is a true streaming filter), then the
same windowing. If the source rate differs from 128 Hz, each emitted window is
resampled to 640 samples (polyphase).
"""
from fractions import Fraction
from typing import List, Optional

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfilt_zi, tf2sos


class StreamingPreprocessor:
    def __init__(self, srate: float, n_channels: int, window_sec: float = 5.0, hop_sec: float = 2.5,
                 target_fs: float = 128.0, bandpass_hz=None, notch_hz=None, remove_dc=False):
        self.srate, self.C = float(srate), n_channels
        self.win = int(round(window_sec * srate)); self.hop = int(round(hop_sec * srate))
        self.target_fs = target_fs; self.target_len = int(round(window_sec * target_fs))
        self.buf = np.zeros((n_channels, 0)); self.n_seen = 0; self.next_start = 0
        sos = []
        if remove_dc:
            sos.append(butter(1, 0.3, btype='high', fs=srate, output='sos'))
        if bandpass_hz:
            sos.append(butter(4, bandpass_hz, btype='band', fs=srate, output='sos'))
        if notch_hz and notch_hz < srate / 2:
            b, a = iirnotch(notch_hz, 30.0, fs=srate); sos.append(tf2sos(b, a))
        self.sos = np.vstack(sos) if sos else None
        self.zi = None

    def push(self, chunk: np.ndarray) -> List[np.ndarray]:
        """chunk: (n_samples, C). Returns the list of completed windows (C, 640) at 128 Hz."""
        if chunk.size == 0:
            return []
        x = chunk.T.astype(np.float64)
        if self.sos is not None:
            if self.zi is None:
                self.zi = np.repeat(sosfilt_zi(self.sos)[:, None, :], self.C, axis=1) * x[None, :, :1]
            x, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        self.buf = np.concatenate([self.buf, x], axis=1)
        self.display = np.concatenate([getattr(self, 'display', np.zeros((self.C, 0))), x],
                                      axis=1)[:, -int(10 * self.srate):]
        self.n_seen += x.shape[1]
        buf_start = self.n_seen - self.buf.shape[1]          # absolute index of buf[:, 0]
        out = []
        while self.next_start + self.win <= self.n_seen:      # a full window is available
            off = self.next_start - buf_start
            out.append(self._resample(self.buf[:, off:off + self.win]))
            self.next_start += self.hop
        cut = self.next_start - buf_start                      # samples no later window needs
        if cut > 0:
            self.buf = self.buf[:, cut:]
        return out

    def recent(self, seconds: float) -> np.ndarray:
        """Most recent filtered samples (for plotting), (C, <= min(seconds, 10) * srate)."""
        d = getattr(self, 'display', np.zeros((self.C, 0)))
        return d[:, -int(seconds * self.srate):]

    def _resample(self, w):
        if w.shape[1] == self.target_len:
            return w.copy()
        fr = Fraction(self.target_len, w.shape[1]).limit_denominator(1000)
        r = resample_poly(w, fr.numerator, fr.denominator, axis=-1)
        return r[:, :self.target_len] if r.shape[1] >= self.target_len else \
            np.pad(r, ((0, 0), (0, self.target_len - r.shape[1])), mode='edge')


def quality(window: np.ndarray, flat_sd: float = 0.5, max_abs: float = 150.0) -> dict:
    """Per-channel flags: 'flat' (SD too small), 'artefact' (|x - median| too large), 'ok'."""
    sd = window.std(1); dev = np.abs(window - np.median(window, axis=1, keepdims=True)).max(1)
    flags = np.where(sd < flat_sd, 'flat', np.where(dev > max_abs, 'artefact', 'ok'))
    return {'flags': flags.tolist(), 'sd': sd.tolist(), 'max_dev': dev.tolist(),
            'fraction_ok': float(np.mean(flags == 'ok'))}
