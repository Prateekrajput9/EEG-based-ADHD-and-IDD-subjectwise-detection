"""Figures built only from saved result JSON files.

Style: plain report figures. White background, thin black axes on all sides,
serif text to match the LaTeX report, light dotted grid, and a small muted
palette (steel blue, brick red, green, grey, ...) used in a fixed order.
"""
import json
from pathlib import Path

import numpy as np

SERIES = ['#4c72a4', '#b9493f', '#5e9a68', '#7f7f7f', '#c49a3a', '#7d6aa8', '#3f8f93', '#a65e8a']
INK, INK2, GRID, SURFACE = '#000000', '#444444', '#d0d0d0', '#ffffff'
LABELS = {'DS1_Rest': 'DS-1 rest', 'DS1_Music': 'DS-1 music',
          'DS1_RestMusic': 'DS-1 rest+music', 'DS2': 'DS-2 ADHD*'}
CMAP = 'Blues'          # sequential colour map for scalp maps


def _rc():
    import matplotlib
    matplotlib.rcParams.update({'font.family': 'serif',
                                'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                                'mathtext.fontset': 'stix', 'axes.titleweight': 'normal'})


_rc()


def style(ax):
    _rc()
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(INK); s.set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=8, width=0.6, direction='out')
    ax.yaxis.grid(True, color=GRID, lw=0.5, ls=':')
    ax.set_axisbelow(True)


def save(fig, base):
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight', facecolor=SURFACE)


def _load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}


def topo(ax, values, ch_names, vmin=None, vmax=None, cmap='Blues', label=None):
    """Scalp map with MNE's standard 10-05 positions (old T3-T6 names mapped)."""
    import mne
    names = [ALIASES.get(c, c) for c in ch_names]
    info = mne.create_info(names, 128.0, 'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'))
    im, _ = mne.viz.plot_topomap(np.asarray(values, float), info, axes=ax, show=False, cmap=cmap,
                                 vlim=(vmin, vmax), names=names if label else None, sensors=True,
                                 contours=0, sphere=(0, 0, 0, 0.105))   # head outline encloses Fp1/Fp2, O1/O2
    return im


def channel_curve_figure(d, base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exps = [e for e in ('DS1_Rest', 'DS1_Music', 'DS2') if e in d]
    fig, axes = plt.subplots(1, len(exps), figsize=(3.3 * len(exps), 3.0), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), exps):
        style(ax)
        c = d[e]['curve']; k = [r['n_channels'] for r in c]
        v = np.array([r['window_accuracy'] for r in c]) * 100
        lo = np.array([r['window_accuracy_ci'][0] for r in c]) * 100
        hi = np.array([r['window_accuracy_ci'][1] for r in c]) * 100
        ax.fill_between(k, lo, hi, color=SERIES[0], alpha=0.15, lw=0)
        ax.plot(k, v, color=SERIES[0], lw=2, marker='o', ms=4, label='window accuracy')
        cv = np.array([r['child_accuracy'] for r in c]) * 100
        ax.plot(k, cv, color=SERIES[1], lw=2, marker='s', ms=4, label='subject accuracy')
        ax.axhline(50, color=INK2, lw=0.8, ls='--')
        ax.set_title(LABELS[e], fontsize=9, color=INK, loc='left')
        ax.set_xlabel('Electrodes kept (ranked in training fold)', fontsize=8, color=INK)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    np.atleast_1d(axes)[0].set_ylabel('Subject-wise accuracy (%)', fontsize=9, color=INK)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); save(fig, base); plt.close(fig)


def topomap_figure(d, base, k=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exps = [e for e in ('DS1_Rest', 'DS1_Music', 'DS2') if e in d]
    fig, axes = plt.subplots(1, len(exps), figsize=(3.0 * len(exps), 3.0))
    for ax, e in zip(np.atleast_1d(axes), exps):
        names = d[e]['channels']; n = d[e]['n_outer_folds']
        f = d[e]['selection_frequency_top_k'][str(k)]
        im = topo(ax, [f[c] / n for c in names], names, 0, 1, label=True)
        ax.set_title(f'{LABELS[e]}: chosen in top {k}', fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.7, label='fraction of outer folds')
    cb.ax.tick_params(labelsize=7)
    save(fig, base); plt.close(fig)


def gap_figure(root, base, model='knn_tableII'):
    """Segment-wise (leaky) vs subject-wise (honest) accuracy, window and child level."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    exps = [e for e in LABELS if (root / e / f'{model}__segment10.json').exists()]
    rows = []
    for e in exps:
        seg = _load(root / e / f'{model}__segment10.json')
        sub_name = 'subject_loso' if e.startswith('DS1') else 'subject_sgkf'
        sub = _load(root / e / f'{model}__{sub_name}.json')
        rows.append((e, seg, sub))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, level, title in ((axes[0], 'window', 'Window level'),
                             (axes[1], 'child', 'Subject level (majority vote)')):
        style(ax)
        x = np.arange(len(rows)); w = 0.38
        for k, (name, key) in enumerate((('Window-wise split', 1),
                                         ('Subject-wise split', 2))):
            vals, lo, hi = [], [], []
            for e, seg, sub in rows:
                r = seg if key == 1 else sub
                v = r['metrics'][level]['accuracy']; ci = r['ci95'][level]['accuracy']
                vals.append(v * 100); lo.append((v - ci[0]) * 100); hi.append((ci[1] - v) * 100)
            xs = x + (k - 0.5) * (w + 0.02)
            ax.bar(xs, vals, w, color=SERIES[k], label=name, edgecolor=SURFACE, lw=1)
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt='none', ecolor=INK2, elinewidth=0.9, capsize=2)
            for xi, v in zip(xs, vals):
                ax.text(xi, 2, f'{v:.1f}', ha='center', va='bottom', fontsize=7, color=SURFACE)
        ax.axhline(50, color=INK2, lw=0.8, ls='--')
        ax.set_xticks(x); ax.set_xticklabels([LABELS[e] for e, _, _ in rows], fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.set_ylim(0, 105)
    axes[0].set_ylabel('Accuracy (%)', color=INK, fontsize=9)
    axes[0].text(len(rows) - 0.5, 51.5, 'chance', fontsize=7, color=INK2, ha='right')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, base)
    plt.close(fig)
