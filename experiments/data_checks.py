"""Data checks: simple properties of the raw recordings that could explain results.

Descriptive only: nothing here trains or evaluates a classifier, so there is no
train/test split to leak across. Numbers quoted in the report come from the JSON
this script writes.

  1. DS-1 participant ages, from the dataset's own metadata (QualitativeData.xlsx).
  2. DS-1 per-recording amplitude and relative band power by group, and how well
     one number per recording separates the groups (AUC over the 14 recordings).
  3. DS-1 recordings whose amplitude is far outside the rest (artefact check).
  4. DS-2 recording length (windows per child) by group: the task ran until the
     child finished, so length could differ between groups.
  5. Exact (Clopper-Pearson) 95% interval when all 14 DS-1 participants are correct.

    python -m experiments.data_checks
Output: results/data_checks/data_checks.json
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score

OUT = Path('results/data_checks/data_checks.json')
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
FS = 128


def ds1_ages():
    import pandas as pd
    df = pd.read_excel('idd/Data/QualitativeData.xlsx')
    col = next(c for c in df.columns if str(c).lower().startswith('age'))
    ages = [int(a) for a in df[col]]
    return {'source': 'idd/Data/QualitativeData.xlsx', 'group': 'IDD',
            'participants': [str(p) for p in df[df.columns[0]]], 'ages_years': ages,
            'min': min(ages), 'max': max(ages),
            'note': 'The dataset files list ages for the 7 IDD participants only; '
                    'control (TDC) ages are not in the files.'}


def ds1_recordings():
    rows = []
    for f in sorted(glob.glob('idd/Data/CleanData/*/*/*.mat')):
        m = sio.loadmat(f)
        a = np.asarray(m[[k for k in m if not k.startswith('__')][0]], dtype=float)  # (14, T)
        fr, p = welch(a, fs=FS, nperseg=256, noverlap=128, axis=-1)
        tot = p[:, (fr >= 1) & (fr < 45)].sum(-1)
        rel = {b: float((p[:, (fr >= lo) & (fr < hi)].sum(-1) / tot).mean())
               for b, (lo, hi) in BANDS.items()}
        name = os.path.basename(f)
        rows.append({'file': name, 'group': 'IDD' if 'IDD' in f else 'TDC',
                     'condition': 'rest' if '_Rest_' in name else 'music',
                     'sd_uv': float(a.std(axis=1).mean()), 'max_abs_uv': float(np.abs(a).max()),
                     'relpow': rel})
    return rows


def ds1_duplicates():
    """Are any two of the 28 cleaned DS-1 recordings the same data?"""
    import itertools
    D = {}
    for f in sorted(glob.glob('idd/Data/CleanData/*/*/*.mat')):
        m = sio.loadmat(f)
        D[os.path.basename(f)] = np.asarray(m[[k for k in m if not k.startswith('__')][0]], dtype=float)
    identical, worst = [], (0.0, None)
    for a, b in itertools.combinations(sorted(D), 2):
        if D[a].shape != D[b].shape:
            continue
        if np.array_equal(D[a], D[b]):
            identical.append([a, b]); continue
        c = abs(float(np.corrcoef(D[a].ravel(), D[b].ravel())[0, 1]))
        if c > worst[0]:
            worst = (c, [a, b])
    return {'n_recordings': len(D), 'identical_pairs': identical,
            'max_abs_correlation': worst[0], 'max_abs_correlation_pair': worst[1]}


def separation(rows, cond):
    r = [x for x in rows if x['condition'] == cond]
    y = np.array([1 if x['group'] == 'IDD' else 0 for x in r])
    out = {}
    for feat in ['sd_uv'] + [f'rel_{b}' for b in BANDS]:
        v = np.array([x['sd_uv'] if feat == 'sd_uv' else x['relpow'][feat[4:]] for x in r])
        out[feat] = {'auc_idd_higher': float(roc_auc_score(y, v)),
                     'idd_mean': float(v[y == 1].mean()), 'tdc_mean': float(v[y == 0].mean()),
                     'idd_range': [float(v[y == 1].min()), float(v[y == 1].max())],
                     'tdc_range': [float(v[y == 0].min()), float(v[y == 0].max())],
                     'groups_do_not_overlap': bool(v[y == 1].min() > v[y == 0].max()
                                                   or v[y == 0].min() > v[y == 1].max())}
    return out


def outliers(rows, factor=3.0):
    sd = np.array([x['sd_uv'] for x in rows])
    med = float(np.median(sd))
    return {'median_sd_uv': med, 'factor': factor,
            'recordings': [{'file': x['file'], 'sd_uv': x['sd_uv'], 'max_abs_uv': x['max_abs_uv'],
                            'times_median': x['sd_uv'] / med}
                           for x in rows if x['sd_uv'] > factor * med]}


def ds2_lengths():
    import csv
    cnt, lab = {}, {}
    with open('data/metadata/ds2_segments.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            cnt[r['subject_id']] = cnt.get(r['subject_id'], 0) + 1
            lab[r['subject_id']] = int(float(r['label']))
    ks = sorted(cnt)
    n = np.array([cnt[k] for k in ks]); y = np.array([lab[k] for k in ks])
    rng = np.random.default_rng(42)
    boot = []
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(2000):
        s = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        boot.append(roc_auc_score(y[s], n[s]))
    return {'source': 'data/metadata/ds2_segments.csv', 'n_children': len(ks),
            'adhd': {'n': int((y == 1).sum()), 'mean_windows': float(n[y == 1].mean()),
                     'min': int(n[y == 1].min()), 'max': int(n[y == 1].max())},
            'control': {'n': int((y == 0).sum()), 'mean_windows': float(n[y == 0].mean()),
                        'min': int(n[y == 0].min()), 'max': int(n[y == 0].max())},
            'auc_length_predicts_adhd': float(roc_auc_score(y, n)),
            'auc_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'ci_method': 'percentile bootstrap over children, stratified by class, 2000 resamples, seed 42'}


def exact_ci(k, n, level=0.95):
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return {'k': k, 'n': n, 'level': level, 'method': 'Clopper-Pearson', 'ci': [lo, hi]}


def main():
    rows = ds1_recordings()
    out = {'ds1_ages': ds1_ages(),
           'ds1_recordings': rows,
           'ds1_separation': {c: separation(rows, c) for c in ('rest', 'music')},
           'ds1_outliers': outliers(rows),
           'ds1_duplicates': ds1_duplicates(),
           'ds2_lengths': ds2_lengths(),
           'exact_ci_all_correct_ds1': exact_ci(14, 14)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    s = out['ds1_separation']
    print(f"DS-1 IDD ages {out['ds1_ages']['min']}-{out['ds1_ages']['max']}")
    for c in ('rest', 'music'):
        b = s[c]['rel_beta']
        print(f"DS-1 {c}: relative beta AUC {b['auc_idd_higher']:.3f}, no overlap {b['groups_do_not_overlap']}")
    for o in out['ds1_outliers']['recordings']:
        print(f"outlier {o['file']}: SD {o['sd_uv']:.1f} uV ({o['times_median']:.1f}x median), max {o['max_abs_uv']:.0f} uV")
    dup = out['ds1_duplicates']
    print(f"DS-1 duplicates: {len(dup['identical_pairs'])} identical pairs among {dup['n_recordings']} "
          f"recordings, max |correlation| {dup['max_abs_correlation']:.3f}")
    L = out['ds2_lengths']
    print(f"DS-2 windows ADHD {L['adhd']['mean_windows']:.1f} vs control {L['control']['mean_windows']:.1f}, "
          f"AUC {L['auc_length_predicts_adhd']:.3f} CI {L['auc_ci95']}")
    print(f"14/14 exact CI {out['exact_ci_all_correct_ds1']['ci']}")
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
