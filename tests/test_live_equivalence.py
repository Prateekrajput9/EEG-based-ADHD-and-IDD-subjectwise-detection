"""A recorded file pushed through the live path must give the offline windows and features."""
import numpy as np
import pytest

from src.live.preprocess import StreamingPreprocessor
from src.live.sources import ReplaySource


def _offline(cfg, subject):
    from src.windows import load_windows
    W, y, g, m = load_windows(cfg, 'DS1_Rest')
    return W[g == subject]


@pytest.mark.parametrize('subject', ['IDD_001', 'TDC_004'])
def test_replay_windows_and_features_match_offline(subject):
    import os
    if not os.path.isdir('idd/Data/CleanData'):
        pytest.skip('DS-1 data not present')
    from src import features_classic as fc
    from src.utils import load_config
    cfg = load_config('configs/config.yaml')
    src = ReplaySource.from_ds1(cfg, subject, 'rest', speed=None, loop=False, chunk=37)
    src.start()
    pre = StreamingPreprocessor(src.srate, len(src.channels), 5.0, 2.5, 128.0)
    live = []
    while not src.finished:
        live += pre.push(src.read())
    off = _offline(cfg, subject)
    # Offline segmentation uses floor((N - 640) / 320) windows (46 for a 120 s
    # recording, to match the paper's counts) and so omits the final complete
    # window; a stream has no end, so it emits that 47th window too.
    assert len(off) == 46 and len(live) == 47
    live = np.stack(live[:46])
    np.testing.assert_allclose(live, off, rtol=1e-6, atol=1e-4)      # offline cache is float32
    for i in (0, 17, 45):
        a = fc.spectral_one(live[i].astype(np.float32)); b = fc.spectral_one(off[i])
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_streaming_filter_is_chunk_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    out = {}
    for chunk in (1, 7, 500):
        p = StreamingPreprocessor(128.0, 3, 5.0, 2.5, 128.0, bandpass_hz=[1, 30], notch_hz=50, remove_dc=True)
        ws = []
        for i in range(0, len(x), chunk):
            ws += p.push(x[i:i + chunk])
        out[chunk] = np.stack(ws)
    np.testing.assert_allclose(out[1], out[500], atol=1e-10)
    np.testing.assert_allclose(out[7], out[500], atol=1e-10)


def test_resampling_to_128():
    p = StreamingPreprocessor(256.0, 2, 5.0, 2.5, 128.0)
    ws = p.push(np.random.default_rng(1).normal(size=(256 * 12, 2)))
    assert all(w.shape == (2, 640) for w in ws) and len(ws) == 3
