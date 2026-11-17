"""Small CNNs for raw EEG windows (CPU-friendly), with a scikit-learn interface.

EEGNet-8,2 (Lawhern et al., J. Neural Eng. 2018) and ShallowConvNet
(Schirrmeister et al., Hum. Brain Mapp. 2017), sized for 128 Hz, 640-sample
windows. Hyper-parameters are the published defaults, not tuned.

TorchWindowClassifier.fit(X, y, groups):
  * holds out ~20% of the TRAINING children (stratified by class, subject-wise)
    as a validation set for early stopping; the test children are never seen
  * per-channel scaling statistics come from the training windows only
  * class-balanced cross-entropy
  * augmentation (training windows only): random circular time shift,
    amplitude scaling 0.9-1.1, Gaussian noise at 5% of channel SD
  * early stopping on validation balanced accuracy (patience 8, max 60 epochs);
    the best epoch's weights are restored
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold


class EEGNet(nn.Module):
    def __init__(self, n_ch, n_t, n_cls=2, F1=8, D=2, F2=16, k1=64, drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_t))).numel()
        self.head = nn.Linear(n, n_cls)

    def forward(self, x):                     # x: (B, C, T)
        z = self.block2(self.block1(x.unsqueeze(1)))
        return self.head(z.flatten(1))


class ShallowConvNet(nn.Module):
    def __init__(self, n_ch, n_t, n_cls=2, F=40, k=25, pool=75, stride=15, drop=0.5):
        super().__init__()
        self.conv_t = nn.Conv2d(1, F, (1, k))
        self.conv_s = nn.Conv2d(F, F, (n_ch, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            n = self._feat(torch.zeros(1, n_ch, n_t)).numel()
        self.head = nn.Linear(n, n_cls)

    def _feat(self, x):
        z = self.bn(self.conv_s(self.conv_t(x.unsqueeze(1))))
        z = torch.log(torch.clamp(self.pool(z * z), min=1e-6))   # square -> mean -> log
        return self.drop(z).flatten(1)

    def forward(self, x):
        return self.head(self._feat(x))


ARCHS = {'eegnet': EEGNet, 'shallow': ShallowConvNet}


def _bal_acc(y, p):
    return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])


class TorchWindowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, arch='eegnet', max_epochs=60, patience=8, batch_size=64, lr=1e-3,
                 weight_decay=1e-3, augment=True, val_fraction=0.2, seed=42, n_threads=2):
        self.arch = arch; self.max_epochs = max_epochs; self.patience = patience
        self.batch_size = batch_size; self.lr = lr; self.weight_decay = weight_decay
        self.augment = augment; self.val_fraction = val_fraction; self.seed = seed
        self.n_threads = n_threads

    def _scale(self, X):
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def _augment(self, xb, rng):
        B, C, T = xb.shape
        shift = rng.integers(0, T, B)
        idx = (np.arange(T)[None, :] + shift[:, None]) % T
        xb = np.take_along_axis(xb, idx[:, None, :].repeat(C, 1), axis=2)
        xb = xb * rng.uniform(0.9, 1.1, (B, 1, 1)).astype(np.float32)
        return xb + rng.normal(0, 0.05, xb.shape).astype(np.float32)

    def fit(self, X, y, groups=None):
        torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        groups = np.asarray(groups) if groups is not None else np.arange(len(y))
        n_val_folds = max(2, int(round(1 / self.val_fraction)))
        tr, va = next(StratifiedGroupKFold(n_val_folds, shuffle=True, random_state=self.seed)
                      .split(X, y, groups))
        assert not set(groups[tr]) & set(groups[va])
        self.mu_ = X[tr].mean(axis=(0, 2), keepdims=True)
        self.sd_ = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8
        Xtr, Xva = self._scale(X[tr]), torch.from_numpy(self._scale(X[va]))
        ytr, yva = y[tr], y[va]
        self.net_ = ARCHS[self.arch](X.shape[1], X.shape[2])
        w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(1)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait, self.history_ = -1.0, None, 0, []
        for epoch in range(self.max_epochs):
            self.net_.train()
            order = rng.permutation(len(ytr))
            for i in range(0, len(order), self.batch_size):
                b = order[i:i + self.batch_size]
                xb = self._augment(Xtr[b], rng) if self.augment else Xtr[b]
                opt.zero_grad()
                loss = loss_fn(self.net_(torch.from_numpy(xb)), torch.from_numpy(ytr[b]))
                loss.backward(); opt.step()
            self.net_.eval()
            with torch.no_grad():   # batched: one pass over a DS-2 fold needs > 2 GB
                pv = np.concatenate([self.net_(Xva[i:i + 256]).argmax(1).numpy()
                                     for i in range(0, len(Xva), 256)])
            score = _bal_acc(yva, pv)
            self.history_.append(score)
            if score > best + 1e-4:
                best, best_state, wait = score, copy.deepcopy(self.net_.state_dict()), 0
                self.best_epoch_ = epoch
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.val_score_ = best
        return self

    def predict_proba(self, X):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out.append(torch.softmax(self.net_(torch.from_numpy(self._scale(X[i:i + 256]))), 1).numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
