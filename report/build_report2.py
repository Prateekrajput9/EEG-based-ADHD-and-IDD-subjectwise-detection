"""Build the numbers, tables and figures of Report 2 (report/report2/).

Report 2 is centred on the main pipeline
    SMVMD common modes -> covariance -> Riemannian tangent space -> LightGBM / XGBoost.
Every number is read from results/ (Stage 2b, 2c, 2d result files). The earlier study's numbers
(starting method, Steps A-J, data checks) are reused from report/final/generated/final_numbers.tex,
which report/build_final.py writes; that file and a few of its tables/figures are copied here.

    python report/build_final.py      # once, for the earlier study's numbers
    python report/build_report2.py
    cd report/report2 && pdflatex BTP_report2 && pdflatex BTP_report2
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.protocol.figures import SERIES, INK, style, save   # noqa: E402

OUT = ROOT / 'report/report2'
GEN, FIG = OUT / 'generated', OUT / 'figures'
R = ROOT / 'results'
FINAL = ROOT / 'report/final'
EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
TAG = {'DS1_Rest': 'DSoneRest', 'DS1_Music': 'DSoneMusic', 'DS1_RestMusic': 'DSoneRestMusic', 'DS2': 'DStwo'}
LAB = {'DS1_Rest': 'DS-1 rest', 'DS1_Music': 'DS-1 music', 'DS1_RestMusic': 'DS-1 rest+music',
       'DS2': 'DS-2 ADHD$^\\ast$'}
BANDS = ['delta', 'theta', 'alpha', 'beta']
GREEK = {'delta': '$\\delta$', 'theta': '$\\theta$', 'alpha': '$\\alpha$', 'beta': '$\\beta$'}
CLF = {'lr': 'LR', 'svm': 'SVM', 'xgb': 'XGBoost', 'lgbm': 'LightGBM'}
MAIN = 'smvmdcm_select_ts_lgbm'          # the main pipeline
macros, missing = [], []


# ------------------------------------------------------------------ helpers
def mac(name, val):
    macros.append(f'\\newcommand{{\\{name}}}{{{val}}}')


def pct(v, d=1):
    return '??' if v is None else f'{100 * v:.{d}f}'


def ci(c, d=1):
    return '??' if not c or c[0] is None else f'{100 * c[0]:.{d}f}--{100 * c[1]:.{d}f}'


def auc(v):
    return '??' if v is None else f'{v:.3f}'


def auc_ci(c):
    return '??' if not c or c[0] is None else f'{c[0]:.3f}--{c[1]:.3f}'


def sch(exp):
    return 'subject_loso' if exp.startswith('DS1') else 'subject_sgkf'


def res(exp, name, stage='stage2c'):
    """Result JSON of one model; Step G (raw LR / SVM) rows live in Stage 2b."""
    if name in ('raw_ts_lr', 'raw_ts_svm'):
        stage, name = 'stage2b', name.replace('raw_ts', 'riemann_ts')
    p = R / stage / exp / f'{name}__{sch(exp)}.json'
    if not p.exists():
        missing.append(str(p.relative_to(ROOT))); return None
    return json.loads(p.read_text(encoding='utf-8'))


def load(rel):
    p = R / rel
    if not p.exists():
        missing.append(str(p.relative_to(ROOT))); return None
    return json.loads(p.read_text(encoding='utf-8'))


def row_vals(d):
    if d is None:
        return dict(bal='??', balci='??', auc='??', aucci='??', subj='??', subjacc='??', subjci='??',
                    sens='??', spec='??', acc='??')
    w, c, k = d['metrics']['window'], d['metrics']['child'], d['ci95']
    return dict(bal=pct(w['balanced_accuracy']), balci=ci(k['window']['balanced_accuracy']),
                auc=auc(w['roc_auc']), aucci=auc_ci(k['window']['roc_auc']),
                subj=f"{c['n_correct']}/{c['n']}", subjacc=pct(c['accuracy']), subjci=ci(k['child']['accuracy']),
                sens=pct(w.get('sensitivity')), spec=pct(w.get('specificity')), acc=pct(w['accuracy']))


def tab(name, cols, head, rows, size='\\small'):
    L = [size, f'\\begin{{tabular}}{{{cols}}}', '\\hline', ' & '.join(head) + ' \\\\', '\\hline'] + rows + \
        ['\\hline', '\\end{tabular}']
    (GEN / f'{name}.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')


def chosen(d):
    """How often each mode set was chosen by the inner CV, e.g. 'all four 10/10'."""
    if d is None or 'folds' not in d:
        return '??'
    names = {'[0, 1, 2, 3]': 'all four', **{f'[{i}]': GREEK[b] for i, b in enumerate(BANDS)}}
    cnt = Counter(str(f['chosen_params'].get('ts__bands')) for f in d['folds'])
    n = len(d['folds'])
    return ', '.join(f'{names.get(k, k)} {v}/{n}' for k, v in cnt.most_common())


# ------------------------------------------------------------------ copied from report/final
def copy_previous():
    src = FINAL / 'generated/final_numbers.tex'
    if not src.exists():
        sys.exit('run report/build_final.py first')
    shutil.copy(src, GEN / 'final_numbers.tex')
    for t in ('tab_gap', 'tab_progression', 'tab_models_DS1_Rest', 'tab_models_DS1_Music',
              'tab_models_DS1_RestMusic', 'tab_models_DS2'):
        shutil.copy(FINAL / f'generated/{t}.tex', GEN / f'{t}.tex')
    for f in ('IITI.png', 'segment_vs_subject_gap.pdf', 'segment_vs_subject_gap.png',
              'smvmd_mode_count_histogram.pdf', 'smvmd_mode_count_histogram.png'):
        if (FINAL / 'figures' / f).exists():
            shutil.copy(FINAL / 'figures' / f, FIG / f)


# ------------------------------------------------------------------ main pipeline
def main_pipeline():
    for exp in EXPS:
        d = res(exp, MAIN); v = row_vals(d); t = TAG[exp]
        for k, mk in [('bal', 'Bal'), ('balci', 'BalCI'), ('auc', 'AUC'), ('aucci', 'AUCCI'), ('subj', 'Subj'),
                      ('subjacc', 'SubjAcc'), ('subjci', 'SubjAccCI'), ('sens', 'Sens'), ('spec', 'Spec'),
                      ('acc', 'Acc')]:
            mac(f'Main{t}{mk}', v[k])
        mac(f'Main{t}Chosen', chosen(d))
        mac(f'Main{t}Runtime', '??' if d is None else f"{d['runtime_sec']:.0f}")
    # main results table, all four settings
    rows = []
    for exp in EXPS:
        v = row_vals(res(exp, MAIN))
        rows.append(f"{LAB[exp]} & {v['bal']} [{v['balci']}] & {v['auc']} & {v['sens']} & {v['spec']} & "
                    f"{v['subj']} & {v['subjacc']} \\\\")
    tab('tab_main', 'lcccccc', ['Setting', 'Bal.\\ acc.\\ [95\\% CI]', 'AUC', 'Sens.', 'Spec.', 'Subjects', 'Subj.\\ acc.'],
        rows)
    # chosen modes per setting
    rows = [f"{LAB[e]} & {chosen(res(e, MAIN))} \\\\" for e in EXPS]
    tab('tab_chosen', 'lp{9cm}', ['Setting', 'Mode set chosen by the inner CV (outer folds)'], rows)


def classifier_ablation():
    """Classifier x representation on DS-2 (and DS-1 rest+music summary)."""
    reps = [('smvmdcm_select', 'SMVMD common modes, mode set chosen in training folds'),
            ('smvmdcm_all', 'SMVMD common modes, all four'),
            ('raw', 'No decomposition (broadband covariance, Step 7 input)')]
    rows = []
    for key, lab in reps:
        rows.append(f'\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\')
        for c in ('lr', 'svm', 'xgb', 'lgbm'):
            v = row_vals(res('DS2', f'{key}_ts_{c}'))
            rows.append(f"\\quad {CLF[c]} & {v['bal']} & {v['auc']} & {v['subj']} \\\\")
            t = {'smvmdcm_select': 'Select', 'smvmdcm_all': 'All', 'raw': 'Raw'}[key] + \
                {'lr': 'Lr', 'svm': 'Svm', 'xgb': 'Xgb', 'lgbm': 'Lgbm'}[c]
            mac(f'Abl{t}Bal', v['bal']); mac(f'Abl{t}BalCI', v['balci']); mac(f'Abl{t}Subj', v['subj'])
            mac(f'Abl{t}AUC', v['auc'])
    tab('tab_classifiers', 'lccc', ['Classifier', 'Bal.\\ acc.\\ (\\%)', 'AUC', 'Children'], rows)
    # figure: balanced accuracy with CI, grouped by classifier
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(4); wdt = 0.26
    for j, (key, lab) in enumerate([('smvmdcm_select', 'SMVMD common modes (chosen in training folds)'),
                                    ('smvmdcm_all', 'SMVMD common modes (all four)'),
                                    ('raw', 'No decomposition')]):
        m, lo, hi = [], [], []
        for c in ('lr', 'svm', 'xgb', 'lgbm'):
            d = res('DS2', f'{key}_ts_{c}')
            if d is None:
                m.append(np.nan); lo.append(0); hi.append(0); continue
            b = d['metrics']['window']['balanced_accuracy']; cc = d['ci95']['window']['balanced_accuracy']
            m.append(100 * b); lo.append(100 * (b - cc[0])); hi.append(100 * (cc[1] - b))
        ax.bar(x + (j - 1) * wdt, m, wdt * 0.92, yerr=[lo, hi], color=SERIES[j], label=lab,
               error_kw=dict(lw=0.6, capsize=2, ecolor=INK))
    ax.axhline(50, color=INK, lw=0.6, ls='--')
    ax.set_xticks(x); ax.set_xticklabels([CLF[c] for c in ('lr', 'svm', 'xgb', 'lgbm')], fontsize=9)
    ax.set_ylabel('Balanced accuracy (%)', fontsize=9); ax.set_ylim(45, 85)
    style(ax); ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=1)
    save(fig, FIG / 'classifier_comparison'); plt.close(fig)


def mode_ablation():
    rows = []
    order = [(f'{b}', GREEK[b] + ' mode only') for b in BANDS] + [('all', 'all four'), ('recon', 'reconstruction (sum of all modes)')]
    for key, lab in order:
        sm = f'smvmdcm_{key}' if key not in ('recon',) else 'smvmd_recon'
        bw = f'fbbw_{key}' if key != 'recon' else None
        cells = []
        for c in ('lr', 'svm'):
            cells.append(row_vals(res('DS2', f'{sm}_ts_{c}'))['bal'])
        for c in ('lr', 'svm'):
            cells.append(row_vals(res('DS2', f'{bw}_ts_{c}'))['bal'] if bw else '--')
        rows.append(f"{lab} & " + ' & '.join(cells) + ' \\\\')
        for c, v in zip(('lr', 'svm'), cells[:2]):
            mac(f"Mode{key.capitalize()}{CLF[c]}Bal", v)
    tab('tab_modes', 'lcccc', ['Component(s)', 'SMVMD, LR', 'SMVMD, SVM', 'Butterworth, LR', 'Butterworth, SVM'], rows)
    # figure: per-mode balanced accuracy DS-2, SMVMD vs Butterworth (SVM)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    keys = BANDS + ['all']; x = np.arange(len(keys)); wdt = 0.38
    for j, (pre, lab) in enumerate([('smvmdcm', 'SMVMD common mode'), ('fbbw', 'Butterworth band')]):
        m, lo, hi = [], [], []
        for k in keys:
            d = res('DS2', f'{pre}_{k}_ts_svm')
            b = d['metrics']['window']['balanced_accuracy'] if d else np.nan
            cc = d['ci95']['window']['balanced_accuracy'] if d else [b, b]
            m.append(100 * b); lo.append(100 * (b - cc[0])); hi.append(100 * (cc[1] - b))
        ax.bar(x + (j - 0.5) * wdt, m, wdt * 0.92, yerr=[lo, hi], color=SERIES[j], label=lab,
               error_kw=dict(lw=0.6, capsize=2, ecolor=INK))
    g = res('DS2', 'raw_ts_svm')
    if g:
        ax.axhline(100 * g['metrics']['window']['balanced_accuracy'], color=SERIES[3], lw=0.9, ls='-',
                   label='No decomposition (Step 7)')
    ax.axhline(50, color=INK, lw=0.6, ls='--')
    ax.set_xticks(x); ax.set_xticklabels([GREEK[k] for k in BANDS] + ['all four'], fontsize=10)
    ax.set_ylabel('Balanced accuracy (%)', fontsize=9); ax.set_ylim(45, 85)
    style(ax); ax.legend(fontsize=7, frameon=False, loc='upper right')
    save(fig, FIG / 'mode_comparison'); plt.close(fig)


def common_modes():
    s = load('stage2c/common_modes_summary.json')
    if s is None:
        return
    rows = []
    for exp, lab in [('DS1_RestMusic', 'DS-1 (rest+music)'), ('DS2', 'DS-2')]:
        d = s[exp]; t = TAG[exp]
        mac(f'Cm{t}ModesMean', f"{d['modes_per_window_mean']:.1f}")
        mac(f'Cm{t}ModesMin', d['modes_per_window_min']); mac(f'Cm{t}ModesMax', d['modes_per_window_max'])
        for b in BANDS:
            mac(f'Cm{t}Empty{b.capitalize()}', f"{d['pct_windows_without_mode'][b]:.1f}")
            mac(f'Cm{t}Share{b.capitalize()}', f"{d['energy_share_pct_mean'][b]:.1f}")
        mac(f'Cm{t}ShareAbove', f"{d['energy_share_pct_mean']['above30']:.2f}")
        rows.append(f"{lab} & {d['modes_per_window_mean']:.1f} ({d['modes_per_window_min']}--{d['modes_per_window_max']}) & "
                    + ' & '.join(f"{d['energy_share_pct_mean'][b]:.1f} / {d['pct_windows_without_mode'][b]:.1f}" for b in BANDS)
                    + ' \\\\')
    tab('tab_common_modes', 'lccccc', ['Data', 'Modes per window', '$\\delta$', '$\\theta$', '$\\alpha$', '$\\beta$'], rows)
    # centre-frequency histogram from the covariance cache
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=False)
    for ax, exp, lab in [(axs[0], 'DS1_RestMusic', 'DS-1 (rest+music)'), (axs[1], 'DS2', 'DS-2')]:
        p = sorted((R / 'cache/smvmd_riemann').glob(f'{exp}_*.npz'))
        if not p:
            missing.append(f'results/cache/smvmd_riemann/{exp}_*.npz'); continue
        cf = np.load(p[0])['centre_hz']; cf = cf[~np.isnan(cf)]
        ax.hist(cf, bins=np.arange(0, 64.5, 1), color=SERIES[0], edgecolor='white', lw=0.3)
        for e in (4, 8, 13, 30):
            ax.axvline(e, color=SERIES[1], lw=0.7, ls='--')
        ax.set_title(lab, fontsize=9); ax.set_xlabel('Mode centre frequency (Hz)', fontsize=8)
        style(ax)
    axs[0].set_ylabel('Number of modes', fontsize=8)
    save(fig, FIG / 'centre_frequencies'); plt.close(fig)


def ds1_table():
    rows = []
    for exp in ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic']:
        for key, lab in [(MAIN, 'Main pipeline (SMVMD common modes + LightGBM)'),
                         ('smvmdcm_select_ts_xgb', 'SMVMD common modes + XGBoost'),
                         ('raw_ts_svm', 'No decomposition + SVM (Step 7)')]:
            v = row_vals(res(exp, key))
            rows.append(f"{LAB[exp] if key == MAIN else ''} & {lab} & {v['bal']} [{v['balci']}] & {v['auc']} & {v['subj']} \\\\")
    tab('tab_ds1', 'llccc', ['Setting', 'Pipeline', 'Bal.\\ acc.\\ [95\\% CI]', 'AUC', 'Participants'], rows)


def literature_ours():
    """Our numbers that appear in the literature comparison table."""
    d = res('DS2', MAIN)
    if d:
        mac('MainDStwoWinAccPlain', pct(d['metrics']['window']['accuracy']))


# ------------------------------------------------------------------ Stage 2d analyses
def permutation():
    p = load('stage2d/permutation_ds1.json')
    if p is None:
        return
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, exp in zip(axs, ['DS1_Rest', 'DS1_Music']):
        d = p[exp]; t = TAG[exp]
        mac(f'Perm{t}N', d['n_permutations'])
        mac(f'Perm{t}Real', pct(d['real_window_bal_acc'])); mac(f'Perm{t}RealSubj', f"{d['real_people_correct']}/14")
        mac(f'Perm{t}Mean', pct(d['perm_window_bal_acc_mean'])); mac(f'Perm{t}Max', pct(d['perm_window_bal_acc_max']))
        mac(f'Perm{t}PNinetyFive', pct(d['perm_window_bal_acc_p95']))
        mac(f'Perm{t}SubjMax', f"{d['perm_people_correct_max']}/14")
        mac(f'Perm{t}SubjMean', f"{d['perm_people_correct_mean']:.1f}")
        mac(f'Perm{t}P', f"{d['p_value_bal_acc']:.3f}")
        ax.hist(100 * np.array(d['perm_window_bal_acc']), bins=np.arange(0, 101, 4), color=SERIES[3],
                edgecolor='white', lw=0.3, label='shuffled labels')
        ax.axvline(100 * d['real_window_bal_acc'], color=SERIES[1], lw=1.4, label='true labels')
        ax.axvline(50, color=INK, lw=0.6, ls='--')
        ax.set_title(LAB[exp], fontsize=9); ax.set_xlabel('Balanced accuracy (%)', fontsize=8)
        style(ax)
    axs[0].set_ylabel('Permutations', fontsize=8); axs[0].legend(fontsize=7, frameon=False, loc='upper left')
    save(fig, FIG / 'permutation_main'); plt.close(fig)


def gap_main():
    g = load('stage2d/gap_main_pipeline.json')
    if g is None:
        return
    rows = []
    for exp in ['DS2', 'DS1_Rest']:
        d = g[exp]; t = TAG[exp]
        lw, sw = d['window_wise_leaky'], d['subject_wise']
        mac(f'Gap{t}Leaky', pct(lw['window_bal_acc'])); mac(f'Gap{t}Subj', pct(sw['window_bal_acc']))
        mac(f'Gap{t}Points', f"{d['gap_bal_acc_points']:.1f}")
        rows.append(f"{LAB[exp]} & {pct(lw['window_bal_acc'])} & {pct(sw['window_bal_acc'])} & "
                    f"{d['gap_bal_acc_points']:.1f} \\\\")
    tab('tab_gap_main', 'lccc', ['Data', 'Window-wise (leaky)', 'Subject-wise', 'Difference (points)'], rows)


def importance():
    d = load('stage2d/importance_main_pipeline.json')
    if d is None:
        return
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={'width_ratios': [1, 2.4]})
    imp = d['DS2']
    for b in BANDS:
        mac(f'ImpDStwo{b.capitalize()}', pct(imp['per_mode'][b]))
        if 'DS1_Rest' in d:
            mac(f'ImpDSoneRest{b.capitalize()}', pct(d['DS1_Rest']['per_mode'][b]))
    mac('ImpDStwoCovShare', pct(imp['share_covariance_terms']))
    mac('ImpDStwoVarShare', pct(imp['share_variance_terms']))
    chs = list(imp['per_channel'])
    mac('ImpDStwoTopChannels', ', '.join(chs[:5]))
    if 'DS1_Rest' in d:
        mac('ImpDSoneRestTopChannels', ', '.join(list(d['DS1_Rest']['per_channel'])[:5]))
    rows = []
    for f in imp['top20_features'][:10]:
        pair = f['ch1'] if f['type'] == 'variance' else f"{f['ch1']}--{f['ch2']}"
        rows.append(f"{GREEK[f['mode']]} & {pair} & {f['type']} & {100 * f['gain_share']:.2f} \\\\")
    tab('tab_top_features', 'llcc', ['Mode', 'Channel(s)', 'Type', 'Gain share (\\%)'], rows)
    axs[0].bar(range(4), [100 * imp['per_mode'][b] for b in BANDS], color=SERIES[0], width=0.6)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels([GREEK[b] for b in BANDS], fontsize=10)
    axs[0].set_ylabel('Share of LightGBM gain (%)', fontsize=8); axs[0].set_title('per mode', fontsize=9)
    vals = [100 * imp['per_channel'][c] for c in chs]
    axs[1].bar(range(len(chs)), vals, color=SERIES[0], width=0.7)
    axs[1].set_xticks(range(len(chs))); axs[1].set_xticklabels(chs, rotation=90, fontsize=7)
    axs[1].set_title('per electrode', fontsize=9)
    for ax in axs:
        style(ax)
    save(fig, FIG / 'importance_main'); plt.close(fig)


def runtime():
    r = load('stage2d/runtime_main_pipeline.json')
    if r is None:
        return
    pw = r['per_window_ms']
    mac('RtSmvmd', f"{pw['smvmd']:.0f}"); mac('RtCov', f"{pw['covariances_4_modes']:.1f}")
    mac('RtPred', f"{pw['tangent_space_plus_lightgbm']:.2f}"); mac('RtTotal', f"{pw['total']:.0f}")
    mac('RtRealtime', f"{5000 / pw['total']:.0f}")
    mac('RtTrain', f"{r['train_seconds_one_model']:.0f}"); mac('RtTrainWin', r['train_windows'])
    mac('RtFeatures', r['n_features']); mac('RtTrees', r['n_trees']); mac('RtLeaves', r['n_leaves_total'])
    mac('RtModelKB', f"{r['pickled_model_kb']:.0f}")


def progression():
    """Bar chart of DS-2 balanced accuracy for Steps 1-10 of the preliminary study and the
    proposed pipeline. Step values are the generated macros of report/build_final.py."""
    import re
    import matplotlib.pyplot as plt
    prev = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}',
                           (GEN / 'final_numbers.tex').read_text(encoding='utf-8')))
    d = res('DS2', MAIN)
    if d is None:
        return
    vals, subj = [], []
    for L in 'ABCDEFGHIJ':
        vals.append(float(prev[f'Step{L}DStwoBal'])); subj.append(prev[f'Step{L}DStwoSubj'].split('/')[0])
    vals.append(100 * d['metrics']['window']['balanced_accuracy']); subj.append(str(d['metrics']['child']['n_correct']))
    labels = [str(i) for i in range(1, 11)] + ['Proposed']
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    cols = [SERIES[0]] * 10 + [SERIES[1]]
    ax.bar(range(11), vals, color=cols, width=0.7)
    for i, (v, s) in enumerate(zip(vals, subj)):
        ax.text(i, v + 0.6, s, ha='center', va='bottom', fontsize=7, color=INK)
    ax.axhline(50, color=INK, lw=0.6, ls='--')
    ax.set_xticks(range(11)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('Step', fontsize=8); ax.set_ylabel('Balanced accuracy (%)', fontsize=8)
    ax.set_ylim(45, 80)
    style(ax)
    save(fig, FIG / 'progression'); plt.close(fig)


def overleaf_zip():
    import zipfile
    z = OUT / 'BTP_report2_overleaf.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p != z and p.suffix in ('.tex', '.pdf', '.png'):
                f.write(p, p.relative_to(OUT))
    print(f'Overleaf bundle: {z.relative_to(ROOT)}')


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    copy_previous()
    main_pipeline(); classifier_ablation(); mode_ablation(); common_modes(); literature_ours()
    gap_main(); importance(); runtime(); progression()   # ds1_table(), permutation() unused
    (GEN / 'r2_numbers.tex').write_text('% generated by report/build_report2.py from results/ -- do not edit\n'
                                        + '\n'.join(macros) + '\n', encoding='utf-8')
    overleaf_zip()
    print(f'{len(macros)} macros -> {GEN}')
    if missing:
        print('MISSING:'); [print('  ', m) for m in sorted(set(missing))]


if __name__ == '__main__':
    main()
