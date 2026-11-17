"""EEG sources with one interface.

    src = ReplaySource(...) or LSLSource(...)
    src.start()
    chunk = src.read()        # (n_new_samples, n_channels) float64, may be empty
    src.channels, src.srate

ReplaySource paces itself by the wall clock (speed 1.0 = real time). With
speed=None it returns the recording in fixed chunks as fast as possible (tests).
"""
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EEGSource(ABC):
    channels: List[str]
    srate: float

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    def stop(self) -> None:
        pass


class ReplaySource(EEGSource):
    def __init__(self, data: np.ndarray, channels: List[str], srate: float = 128.0,
                 speed: Optional[float] = 1.0, loop: bool = True, chunk: int = 16):
        """data: (n_channels, n_samples) recording."""
        self.data = np.asarray(data, dtype=np.float64)
        self.channels = list(channels); self.srate = float(srate)
        self.speed = speed; self.loop = loop; self.chunk = chunk
        self.pos = 0; self.t0 = None

    @classmethod
    def from_ds1(cls, cfg, subject: str, condition: str = 'rest', **kw):
        from src.data_loader import DS1Loader
        from src.preprocessing import preprocess_ds1
        ds1 = cfg['ds1']
        loader = DS1Loader(cfg['paths']['ds1_data_root'], mode=ds1['preprocessing']['mode'])
        files = loader.discover_files()
        eeg, _ = loader.load_subject(subject, condition, files[subject][condition])
        eeg = preprocess_ds1(eeg, fs=ds1['sampling_freq'], mode=ds1['preprocessing']['mode'],
                             config=ds1['preprocessing'])
        return cls(eeg, ds1['channel_names'], ds1['sampling_freq'], **kw)

    def start(self):
        self.pos = 0; self.t0 = time.perf_counter()

    @property
    def finished(self) -> bool:
        return not self.loop and self.pos >= self.data.shape[1]

    def read(self) -> np.ndarray:
        n_total = self.data.shape[1]
        if self.speed is None:
            due = self.pos + self.chunk
        else:
            due = int((time.perf_counter() - self.t0) * self.srate * self.speed)
        out = []
        while self.pos < due:
            if self.pos >= n_total:
                if not self.loop:
                    break
                due -= n_total; self.pos = 0
                self.t0 = None if self.speed is None else self.t0 + n_total / (self.srate * self.speed)
            take = min(due, n_total) - self.pos
            out.append(self.data[:, self.pos:self.pos + take]); self.pos += take
        return np.concatenate(out, axis=1).T if out else np.zeros((0, len(self.channels)))


class LSLSource(EEGSource):
    """Lab Streaming Layer inlet (pip install pylsl). Channels are matched by label."""

    def __init__(self, stream_type='EEG', stream_name=None, channels=None, resolve_timeout_sec=10):
        self.stream_type = stream_type; self.stream_name = stream_name
        self.wanted = list(channels) if channels else None
        self.timeout = resolve_timeout_sec

    def start(self):
        from pylsl import StreamInlet, resolve_byprop
        prop, val = ('name', self.stream_name) if self.stream_name else ('type', self.stream_type)
        streams = resolve_byprop(prop, val, timeout=self.timeout)
        if not streams:
            raise RuntimeError(f'No LSL stream with {prop}={val!r} found within {self.timeout}s. '
                               'Is the headset streaming (EmotivPRO -> Data Streams -> LSL)?')
        self.inlet = StreamInlet(streams[0], max_chunklen=32)
        info = self.inlet.info()
        self.srate = float(info.nominal_srate())
        labels, ch = [], info.desc().child('channels').child('channel')
        for _ in range(info.channel_count()):
            labels.append(ch.child_value('label')); ch = ch.next_sibling()
        if self.wanted:
            missing = [c for c in self.wanted if c not in labels]
            if missing and len(labels) >= len(self.wanted) and not any(labels):
                self.index = list(range(len(self.wanted)))          # unlabelled stream: assume order
            elif missing:
                raise RuntimeError(f'LSL stream lacks channels {missing}; it has {labels}')
            else:
                self.index = [labels.index(c) for c in self.wanted]
            self.channels = self.wanted
        else:
            self.index = list(range(len(labels))); self.channels = labels

    def read(self) -> np.ndarray:
        samples, _ = self.inlet.pull_chunk(timeout=0.0)
        if not samples:
            return np.zeros((0, len(self.channels)))
        return np.asarray(samples, dtype=np.float64)[:, self.index]
