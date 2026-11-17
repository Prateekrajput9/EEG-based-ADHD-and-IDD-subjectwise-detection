"""Build every number, table and figure of the final report from results/ only.

    python report/build_final.py

Writes report/final/generated/*.tex (macros + tables) and copies figures to
report/final/figures/. Nothing is typed by hand: if a result file is missing,
the corresponding macro expands to '??' and a warning is printed.
"""
import json
import shutil
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'report/final'
GEN, FIG = OUT / 'generated', OUT / 'figures'
R = ROOT / 'results'
EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
TAG = {'DS1_Rest': 'DSoneRest', 'DS1_Music': 'DSoneMusic', 'DS1_RestMusic': 'DSoneRestMusic', 'DS2': 'DStwo'}
LAB = {'DS1_Rest': 'DS-1 rest', 'DS1_Music': 'DS-1 music', 'DS1_RestMusic': 'DS-1 rest+music',
       'DS2': 'DS-2 ADHD$^\\ast$'}
macros, missing = [], []


def load(p):
    p = R / p
    if not p.exists():
        missing.append(str(p.relative_to(ROOT))); return None
    return json.loads(p.read_text(encoding='utf-8'))


def pct(v, d=1):
    return '??' if v is None else f'{100 * v:.{d}f}'


def ci(c, d=1):     # same precision as the point estimates, so 99.5 never prints as 100
    return '??' if not c or c[0] is None else f'{100 * c[0]:.{d}f}--{100 * c[1]:.{d}f}'


def mac(name, val):
    macros.append(f'\\newcommand{{\\{name}}}{{{val}}}')


def sub_name(exp):
    return 'subject_loso' if exp.startswith('DS1') else 'subject_sgkf'


def gap():
    rows = []
    for e in EXPS:
        s = load(f'stage1_eval/{e}/knn_tableII__segment10.json')
        u = load(f'stage1_eval/{e}/knn_tableII__{sub_name(e)}.json')
        if not (s and u):
            continue
        t = TAG[e]
        for k, d in (('Seg', s), ('Subj', u)):
            mac(f'{t}{k}WinAcc', pct(d['metrics']['window']['accuracy'], 2))
            mac(f'{t}{k}WinAccCI', ci(d['ci95']['window']['accuracy']))
            mac(f'{t}{k}ChildCorrect', f"{d['metrics']['child']['n_correct']}/{d['metrics']['child']['n']}")
            mac(f'{t}{k}ChildAccCI', ci(d['ci95']['child']['accuracy']))
            mac(f'{t}{k}WinAUC', f"{d['metrics']['window']['roc_auc']:.3f}")
        rows.append(f"{LAB[e]} & {pct(s['metrics']['window']['accuracy'])} & "
                    f"{pct(u['metrics']['window']['accuracy'])} [{ci(u['ci95']['window']['accuracy'])}] & "
                    f"{u['metrics']['window']['roc_auc']:.3f} & "
                    f"{u['metrics']['child']['n_correct']}/{u['metrics']['child']['n']} "
                    f"[{ci(u['ci95']['child']['accuracy'])}] \\\\")
    old = load('stage1_eval/DS2/knn_tableII__subject_gkf.json')
    if old:
        mac('DStwoGkfWinAcc', pct(old['metrics']['window']['accuracy'], 2))
        mac('DStwoGkfChildCorrect', f"{old['metrics']['child']['n_correct']}/{old['metrics']['child']['n']}")
    sv = load('stage1_eval/DS2/knn_tableII__split_seed_variance.json')
    if sv:
        w = sv['window_accuracy']
        mac('DStwoSplitMean', pct(w['mean'])); mac('DStwoSplitSD', pct(w['std']))
        mac('DStwoSplitMin', pct(w['min'])); mac('DStwoSplitMax', pct(w['max']))
    for e in EXPS:
        n = load(f'stage1_eval/{e}/knn_tuned__{sub_name(e)}.json')
        if n:
            t = TAG[e]
            mac(f'{t}TunedWinAcc', pct(n['metrics']['window']['accuracy']))
            mac(f'{t}TunedWinAccCI', ci(n['ci95']['window']['accuracy']))
            mac(f'{t}TunedChildCorrect', f"{n['metrics']['child']['n_correct']}/{n['metrics']['child']['n']}")
    # Published values of Chandela et al. (2025) -- cited, not produced by this code.
    mac('PaperDSoneAcc', '100'); mac('PaperDStwoAcc', '99.17')
    tab('tab_gap', 'lrrrr', ['Setting', 'Window-wise acc.', 'Subject-wise acc. [95\\% CI]',
                             'Subj.\\ AUC', 'Subjects correct [CI]'], rows)


def tab(name, cols, head, rows, size='\\small'):
    L = [f'{size}', f'\\begin{{tabular}}{{{cols}}}', '\\hline', ' & '.join(head) + ' \\\\', '\\hline'] + rows + \
        ['\\hline', '\\end{tabular}']
    (GEN / f'{name}.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')


def esc(s):
    return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')


def constant(r):
    """True if the saved out-of-fold predictions are one class for every window."""
    p = ROOT / r['source'].replace('.json', '.npz')
    return p.exists() and len(np.unique(np.load(p)['y_pred'])) == 1


FEATURE_NAMES = {'fb_logvar': 'filter-bank log-variance', 'wpt': 'wavelet-packet energies',
                 'aperiodic': 'aperiodic 1/f + entropies', 'SMVMD': 'SMVMD features',
                 'MiniRocket 10k kernels (sktime)': 'MiniRocket'}
# Neutral names for the result files' model labels (the starting method's KNN settings come
# from the literature; see the report, Section on the SMVMD method).
MODEL_RENAME = {'SMVMD+mRMR+KNN (paper Table II)': 'mRMR + KNN (literature settings)',
                'SMVMD+mRMR+KNN (nested-tuned)': 'mRMR + KNN (tuned)',
                'MiniRocket + ridge': 'ridge classifier'}


def all_rows():
    """Every subject-wise method result on the common folds, with neutral model names."""
    from src.protocol.tables import collect
    rows = collect([str(R / 'stage1_eval'), str(R / 'stage2'), str(R / 'stage2b')])
    # the unstratified DS-2 split (an earlier-stage number) uses different folds from every
    # other method; it is reported on its own (DStwoGkf*), not in the same-splits comparison
    rows = [r for r in rows if r['protocol'] != 'subject_gkf']
    for r in rows:
        r['model'] = MODEL_RENAME.get(r['model'], r['model'])
    return rows


def method_name(r):
    """'features + classifier', without repeating words the two fields share."""
    f, m = FEATURE_NAMES.get(r['features'], r['features']), r['model']
    if f.lower() in m.lower() or f.startswith('members'):
        return m
    shared = {w for w in f.lower().split() if len(w) > 3} & {w for w in m.lower().split() if len(w) > 3}
    if shared and '+' in m:
        return f"{f} + {m.split('+', 1)[1].strip()}"
    return f'{f} + {m}'


def models():
    rows = all_rows()
    if not rows:
        missing.append('results/stage2*'); return
    best = {}
    for e in EXPS:
        rs = [r for r in rows if r['experiment'] == e]
        if not rs:
            continue
        L = []
        for r in rs:
            flag = ('$^\\dagger$' if r['transductive'] else '') + ('$^\\ddagger$' if constant(r) else '')
            auc = '--' if r['window_auc'] is None else f"{r['window_auc']:.3f}"
            L.append(f"{esc(FEATURE_NAMES.get(r['features'], r['features']))} & {esc(r['model'])}{flag} & {pct(r['window_acc'])} "
                     f"[{ci(r['window_acc_ci'])}] & {pct(r['window_bal_acc'])} & {auc} & "
                     f"{r['child_correct']}/{r['n_children']} [{ci(r['child_acc_ci'])}] \\\\")
        tab(f'tab_models_{e}', 'llrrrr', ['Features', 'Model', 'Win.\\ acc.\\ [CI]', 'Win.\\ bal.', 'Win.\\ AUC',
                                          'Subjects [CI]'], L, size='\\scriptsize')
        ind = [r for r in rs if not r['transductive']]
        b = max(ind, key=lambda r: r['window_bal_acc'])
        best[e] = b
        t = TAG[e]
        mac(f'{t}BestModel', esc(method_name(b)))
        mac(f'{t}BestWinBal', pct(b['window_bal_acc'])); mac(f'{t}BestWinBalCI', ci(b['window_bal_acc_ci']))
        mac(f'{t}BestChildCorrect', f"{b['child_correct']}/{b['n_children']}")
        mac(f'{t}NModels', str(len(rs)))
        mac(f'{t}NAboveChanceCI', str(sum(1 for r in ind if r['window_bal_acc_ci'] and r['window_bal_acc_ci'][0]
                                          and r['window_bal_acc_ci'][0] > 0.5)))
        mac(f'{t}NInductive', str(len(ind)))
        mac(f'{t}NTransductive', str(len(rs) - len(ind)))
        tc = max(ind, key=lambda r: (r['child_correct'], r['window_bal_acc']))
        mac(f'{t}TopChildModel', esc(method_name(tc)))
        mac(f'{t}TopChildCorrect', f"{tc['child_correct']}/{tc['n_children']}")
        mac(f'{t}TopChildWinBal', pct(tc['window_bal_acc']))
    (GEN / 'best_models.json').write_text(json.dumps(best, indent=1), encoding='utf-8')


def ladder():
    rows = []
    for e in EXPS:
        cells = []
        for rung in ('raw', 'bands', 'vmd', 'mvmd', 'smvmd'):
            d = load(f'stage3/ladder/{e}/{rung}_knnTableII__{sub_name(e)}.json')
            l = load(f'stage3/ladder/{e}/{rung}_lr__{sub_name(e)}.json')
            for m, x in (('Knn', d), ('Lr', l)):
                if x:
                    mac(f'{TAG[e]}Ladder{rung.capitalize()}{m}', pct(x['metrics']['window']['balanced_accuracy']))
            cells.append(f"{pct(d['metrics']['window']['balanced_accuracy']) if d else '??'} / "
                         f"{pct(l['metrics']['window']['balanced_accuracy']) if l else '??'}")
        rows.append(f'{LAB[e]} & ' + ' & '.join(cells) + ' \\\\')
    tab('tab_ladder', 'lrrrrr', ['Setting', 'Raw', 'Fixed bands', 'VMD', 'MVMD', 'SMVMD'], rows)
    m = load('stage3/mrmr_size_curve.json')
    if m:
        L = []
        for e, d in m.items():
            if e not in TAG:
                continue
            L.append(f'{LAB[e]} & ' + ' & '.join(f"{pct(r['window_accuracy'])}" for r in d['curve']) + ' \\\\')
        ns = [str(r['n_features']) for r in next(iter(m.values()))['curve']]
        tab('tab_mrmr', 'l' + 'r' * len(ns), ['Setting'] + ns, L)
    c = load('stage3/cross_condition.json')
    if c:
        L = []
        for k, v in c['results'].items():
            f, a, b = k.split('|')
            f = {'spectral_lr': 'Spectral + LR', 'smvmd_knnTableII': 'SMVMD + mRMR + KNN'}.get(f, f)
            L.append(f"{esc(f)} & {a[6:]} & {b[5:]} & {pct(v['window']['accuracy'])} [{ci(v['ci95']['window']['accuracy'])}]"
                     f" & {v['child']['n_correct']}/{v['child']['n']} \\\\")
        tab('tab_condition', 'lllrr', ['Model', 'Train', 'Test', 'Win.\\ acc.\\ [CI]', 'Subjects'], L)
    rows = []
    for e in ('DS1_Rest', 'DS1_Music', 'DS2'):
        cells = []
        for win, ovl in ((2.5, 50), (5, 50), (10, 50), (5, 0), (5, 75)):
            d = load(f'stage3/window/{e}/spectral_lr_win{win:g}s_ovl{ovl}__{sub_name(e)}.json')
            cells.append(pct(d['metrics']['window']['balanced_accuracy']) if d else '??')
        rows.append(f'{LAB[e]} & ' + ' & '.join(cells) + ' \\\\')
    tab('tab_window', 'lrrrrr', ['Setting', '2.5 s', '5 s', '10 s', '5 s, 0\\%', '5 s, 75\\%'], rows)


DSNAME = {'DS1_Rest': 'DS-1 rest', 'DS1_Music': 'DS-1 music', 'DS2': 'DS-2'}


def cross_label(k):
    """Readable row label for a results key of stage4/cross_dataset.json."""
    p = k.split('|')
    if p[0].startswith('train_'):
        how = 'training-set scaling' if p[2] == 'train_stats' else 'test-set scaling'
        return f"{DSNAME[p[0][6:]]} $\\rightarrow$ {DSNAME[p[1][5:]]} ({how})"
    if p[0].startswith('within_'):
        return f"Within {DSNAME[p[0][7:]]} (subject-wise)"
    a, b = p[1].split('_vs_')
    return f"Dataset identity: {DSNAME[a]} vs {DSNAME[b]}"


def cross():
    d = load('stage4/cross_dataset.json')
    if d:
        L = []
        for k, v in d['results'].items():
            L.append(f"{cross_label(k)} & {pct(v['window']['accuracy'])} & {pct(v['window']['balanced_accuracy'])} & "
                     f"{v['window']['roc_auc']:.3f} [{v['ci95']['window']['roc_auc'][0]:.3f}--{v['ci95']['window']['roc_auc'][1]:.3f}] \\\\")
        tab('tab_cross', 'lrrr', ['Train $\\rightarrow$ test', 'Win.\\ acc.', 'Win.\\ bal.', 'Win.\\ AUC [CI]'], L,
            size='\\scriptsize')
        for a in ('DS1_Rest', 'DS1_Music'):
            v = d['results'].get(f'dataset_id|{a}_vs_DS2')
            if v:
                mac(f'DatasetID{TAG[a]}BalAcc', pct(v['window']['balanced_accuracy']))
        mac('SharedChannels', ', '.join(d['channel_map']['shared_1020']))
        for name, key in (('CrossOneToTwo', 'train_DS1_Rest|test_DS2|train_stats'),
                          ('CrossTwoToOne', 'train_DS2|test_DS1_Rest|train_stats'),
                          ('CrossTwoToOneMusic', 'train_DS2|test_DS1_Music|train_stats'),
                          ('CrossWithinOne', 'within_DS1_Rest|subject_loso'),
                          ('CrossWithinTwo', 'within_DS2|subject_sgkf')):
            v = d['results'].get(key)
            if v:
                mac(f'{name}Bal', pct(v['window']['balanced_accuracy']))
                mac(f'{name}AUC', f"{v['window']['roc_auc']:.3f}")
                mac(f'{name}AUCCI', '--'.join(f'{x:.3f}' for x in v['ci95']['window']['roc_auc']))
    c = load('stage4/channel_reduction.json')
    if c:
        for e in ('DS1_Rest', 'DS1_Music', 'DS2'):
            if e in c:
                cu = c[e]['curve']
                mac(f'{TAG[e]}ChanAll', pct(cu[-1]['window_accuracy']))
                mac(f'{TAG[e]}ChanAllCI', ci(cu[-1]['window_accuracy_ci']))
                for k in (1, 4):
                    w = "One" if k == 1 else "Four"
                    mac(f'{TAG[e]}Chan{w}', pct(cu[k - 1]['window_accuracy']))
                    mac(f'{TAG[e]}Chan{w}CI', ci(cu[k - 1]['window_accuracy_ci']))


def explain():
    s = load('stage5/shap_xgb_spectral.json')
    if s:
        for e in ('DS1_Rest', 'DS1_Music', 'DS2'):
            if e in s:
                top = sorted(s[e]['by_feature'].items(), key=lambda kv: -kv[1])[:3]
                mac(f'{TAG[e]}ShapTop', esc(', '.join(k for k, _ in top)))
                topc = sorted(s[e]['by_channel'].items(), key=lambda kv: -kv[1])[:3]
                mac(f'{TAG[e]}ShapTopChan', ', '.join(k for k, _ in topc))
    true = load('stage2/DS1_Music/raw_shallow__subject_loso.json')
    if true:
        mac('ShallowMusicWinAcc', pct(true['metrics']['window']['accuracy']))
    accs = []
    for k in range(2):
        p = load(f'stage2_sanity/DS1_Music/raw_shallow_perm{k}__subject_loso.json')
        if p:
            accs.append(p['metrics']['window']['accuracy'])
    if accs:
        mac('PermMusicWinAccRange', '--'.join(pct(a) for a in sorted(accs)))
    t = load('stage5/tbr_direct_score.json')
    if t:
        for e in EXPS:
            if e in t:
                mac(f'{TAG[e]}TbrAUC', f"{t[e]['window_auc']:.2f}")
                mac(f'{TAG[e]}TbrChildAUC', f"{t[e]['child_auc']:.2f}")
                mac(f'{TAG[e]}TbrAUCCI', '--'.join(f'{v:.2f}' for v in t[e]['window_auc_ci']))
                mac(f'{TAG[e]}TbrChildAUCCI', '--'.join(f'{v:.2f}' for v in t[e]['child_auc_ci']))
    a = load('stage5/eegnet_attribution.json')
    if a:
        for e in ('DS1_Rest', 'DS2'):
            if e in a:
                mac(f'{TAG[e]}EegnetPTrue', f"{a[e]['mean_p_true_unperturbed']:.2f}")


# The methods in the order they were tried; each step is the best inductive variant of its family.
STEPS = [('A', 'SMVMD features + mRMR + KNN (starting method)', lambda r: r['features'] == 'SMVMD' and 'literature' in r['model']),
         ('B', 'Same, with tuned settings', lambda r: r['features'] == 'SMVMD' and 'tuned' in r['model']),
         ('C', 'Spectral features', lambda r: r['features'] == 'spectral'),
         ('D', 'Connectivity features', lambda r: r['features'] == 'connectivity'),
         ('E', 'Filter bank, wavelet packet, aperiodic', lambda r: r['features'] in ('fb_logvar', 'wpt', 'aperiodic')),
         ('F', 'Subject-level aggregation', lambda r: r['features'].startswith('spectral mean')),
         ('G', 'Riemannian tangent space', lambda r: r['features'].startswith('Riemannian')),
         ('H', 'EEGNet, ShallowConvNet', lambda r: r['features'] == 'raw windows'),
         ('I', 'MiniRocket', lambda r: r['features'].startswith('MiniRocket')),
         ('J', 'Ensemble', lambda r: r['features'].startswith('members'))]


def study_tables():
    """Summary, progression and detailed-metric tables for the report."""
    rows = all_rows()
    if not rows:
        return
    summ, det = [], []
    for e in EXPS:
        rs = [r for r in rows if r['experiment'] == e and not r['transductive']]
        base = load(f'stage1_eval/{e}/knn_tableII__{sub_name(e)}.json')
        if not (rs and base):
            continue
        b = max(rs, key=lambda r: r['window_bal_acc'])
        bj = json.loads((ROOT / b['source']).read_text(encoding='utf-8'))
        bc = base['metrics']['child']
        summ.append(f"{LAB[e]} & {pct(base['metrics']['window']['balanced_accuracy'])} "
                    f"[{ci(base['ci95']['window']['balanced_accuracy'])}] & {bc['n_correct']}/{bc['n']} & "
                    f"{esc(method_name(b))} & {pct(b['window_bal_acc'])} [{ci(b['window_bal_acc_ci'])}] & "
                    f"{b['child_correct']}/{b['n_children']} \\\\")
        for name, d in (('SMVMD features + mRMR + KNN (starting method)', base), (esc(method_name(b)), bj)):
            w, c = d['metrics']['window'], d['metrics']['child']
            (tn, fp), (fn, tp) = c['confusion_matrix']
            det.append(f"{LAB[e]} & {name} & {pct(w['accuracy'])} & {pct(w['balanced_accuracy'])} & "
                       f"{pct(w['sensitivity'])} & {pct(w['specificity'])} & {pct(w['f1'])} & "
                       f"{w['roc_auc']:.3f} & {tp}/{fn}/{fp}/{tn} \\\\")
        t = TAG[e]
        mac(f'{t}SubjWinBal', pct(base['metrics']['window']['balanced_accuracy']))
        mac(f'{t}SubjWinBalCI', ci(base['ci95']['window']['balanced_accuracy']))
        mac(f'{t}SubjSens', pct(base['metrics']['window']['sensitivity']))
        mac(f'{t}SubjSpec', pct(base['metrics']['window']['specificity']))
        mac(f'{t}BestChildAcc', pct(bj['metrics']['child']['accuracy']))
        mac(f'{t}BestWinAUC', f"{bj['metrics']['window']['roc_auc']:.3f}")
        mac(f'{t}BestSens', pct(bj['metrics']['window']['sensitivity']))
        mac(f'{t}BestSpec', pct(bj['metrics']['window']['specificity']))
    tab('tab_summary', 'lrrp{3.8cm}rr',
        ['Setting', 'Starting method [CI]', 'Subjects', 'Best method', 'Best [CI]', 'Subjects'],
        summ, size='\\scriptsize')
    tab('tab_detail', 'lp{3.9cm}rrrrrrr',
        ['Setting', 'Model', 'Acc.', 'Bal.', 'Sens.', 'Spec.', 'F1', 'AUC', 'TP/FN/FP/TN'], det,
        size='\\scriptsize')
    # progression: one row per step, best inductive variant in each setting
    L = []
    for letter, name, test in STEPS:
        cells = []
        for e in EXPS:
            v = [r for r in rows if r['experiment'] == e and not r['transductive'] and test(r)]
            if not v:
                cells.append('--'); continue
            b = max(v, key=lambda r: r['window_bal_acc'])
            cells.append(pct(b['window_bal_acc']))
            t = TAG[e]
            mac(f'Step{letter}{t}Bal', pct(b['window_bal_acc']))
            mac(f'Step{letter}{t}BalCI', ci(b['window_bal_acc_ci']))
            mac(f'Step{letter}{t}Subj', f"{b['child_correct']}/{b['n_children']}")
            mac(f'Step{letter}{t}Model', esc(method_name(b)))
        d2 = [r for r in rows if r['experiment'] == 'DS2' and not r['transductive'] and test(r)]
        subj = max(d2, key=lambda r: r['window_bal_acc']) if d2 else None
        L.append(f'{letter} & {esc(name)} & ' + ' & '.join(cells) +
                 f" & {subj['child_correct'] if subj else '--'}/{subj['n_children'] if subj else '--'} \\\\")
    tab('tab_progression', 'clrrrrr', ['', 'Method', 'DS-1 rest', 'DS-1 music', 'DS-1 rest+music',
                                       'DS-2 ADHD$^\\ast$', 'DS-2 children'], L)


def dataset_stats():
    s = load('stage1/results_summary.json')
    if not s:
        return
    dc = s.get('dataset_counts', {})
    for e in EXPS:
        if e in dc:
            d, t = dc[e], TAG[e]
            mac(f'{t}NWin', str(d['segments_total'])); mac(f'{t}NWinCtrl', str(d['segments_control']))
            mac(f'{t}NWinDis', str(d['segments_disorder']))
            mac(f'{t}WinPerSubjMin', str(d['segments_per_subject_min']))
            mac(f'{t}WinPerSubjMax', str(d['segments_per_subject_max']))
    for e, t in (('DS1_RestMusic', 'DSone'), ('DS2', 'DStwo')):
        m = s.get('mode_counts', {}).get(e)
        if m:
            o = m['overall']
            mac(f'{t}ModesMean', f"{o['mean']:.1f}"); mac(f'{t}ModesMedian', f"{o['median']:.0f}")
            mac(f'{t}ModesMin', str(o['min'])); mac(f'{t}ModesMax', str(o['max']))
            mac(f'{t}ModesCapPct', f"{o['pct_at_cap_20']:.1f}")


def feature_figures():
    """SMVMD mode-count histogram and t-SNE of the SMVMD features, from results/features/.
    The t-SNE embedding (seeded) is cached in generated/tsne_embedding.npz."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.protocol.figures import INK, INK2, LABELS, SERIES, save, style
    F = R / 'features'
    sets = {'DS1_RestMusic': 'DS-1 (rest and music)', 'DS2': 'DS-2'}
    if all((F / f'{e}_features.npz').exists() for e in sets):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
        for ax, (e, title) in zip(axes, sets.items()):
            z = np.load(F / f'{e}_features.npz', allow_pickle=True)
            k, y = z['n_modes'].astype(int), z['y'].astype(int)
            bins = np.arange(k.min() - 0.5, k.max() + 1.5)
            style(ax)
            for lab, name, c in ((0, 'control', SERIES[0]), (1, 'disorder', SERIES[1])):
                ax.hist(k[y == lab], bins=bins, density=True, histtype='step', lw=1.4, color=c, label=name)
            ax.set_title(title, fontsize=9, color=INK, loc='left')
            ax.set_xlabel('number of SMVMD modes per window', fontsize=8, color=INK)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axes[0].set_ylabel('fraction of windows', fontsize=8, color=INK)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); save(fig, FIG / 'smvmd_mode_count_histogram'); plt.close(fig)
    exps = [e for e in ('DS1_Rest', 'DS2') if (F / f'{e}_features.npz').exists()]
    if not exps:
        return
    cache = GEN / 'tsne_embedding.npz'
    emb = dict(np.load(cache)) if cache.exists() else {}
    from sklearn.preprocessing import StandardScaler
    for e in exps:
        if e not in emb:
            from sklearn.manifold import TSNE
            X = StandardScaler().fit_transform(np.load(F / f'{e}_features.npz')['X'])
            emb[e] = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(X)
    np.savez(cache, **emb)
    fig, axes = plt.subplots(len(exps), 2, figsize=(9, 4.2 * len(exps)))
    for i, e in enumerate(exps):
        z = np.load(F / f'{e}_features.npz', allow_pickle=True)
        y, g, E = z['y'].astype(int), z['groups'].astype(str), emb[e]
        ax = axes[i, 0]; style(ax); ax.yaxis.grid(False)
        for lab, name, c in ((0, 'control', SERIES[0]), (1, 'disorder', SERIES[1])):
            ax.scatter(E[y == lab, 0], E[y == lab, 1], s=3, color=c, alpha=0.6, lw=0, label=name)
        ax.set_title(f'{LABELS[e]}: coloured by class', fontsize=9, color=INK, loc='left')
        ax.legend(frameon=False, fontsize=8, markerscale=3)
        ax = axes[i, 1]; style(ax); ax.yaxis.grid(False)
        subs = sorted(set(g)); cmap = plt.get_cmap('tab20')
        for j, s in enumerate(subs):
            m = g == s
            ax.scatter(E[m, 0], E[m, 1], s=3, color=cmap(j % 20), alpha=0.7, lw=0)
        ax.set_title(f'{LABELS[e]}: coloured by subject ({len(subs)} subjects)', fontsize=9, color=INK, loc='left')
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, FIG / 'tsne_class_vs_subject'); plt.close(fig)


def study_figures():
    """All-methods forest plot and permutation-test figure, written to report/final/figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.protocol.figures import INK, INK2, LABELS, SERIES, SURFACE, save, style
    from src.protocol.tables import collect
    FIG.mkdir(parents=True, exist_ok=True)
    rows = all_rows()
    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(11, 11.5))
        for ax, e in zip(axes.flat, EXPS):
            style(ax); ax.yaxis.grid(False); ax.xaxis.grid(True, color='#e4e3df', lw=0.6)
            rs = sorted([r for r in rows if r['experiment'] == e], key=lambda r: r['window_bal_acc'])
            for i, r in enumerate(rs):
                v = 100 * r['window_bal_acc']; lo, hi = (100 * x for x in r['window_bal_acc_ci'])
                c = SERIES[1] if r['features'] == 'SMVMD' else SERIES[0]
                ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle='round')
                ax.plot(v, i, 'o', ms=6, mfc='white' if r['transductive'] else c, mec=c, mew=1.6)
            ax.set_yticks(range(len(rs)))
            ax.set_yticklabels([method_name(r) + (' †' if r['transductive'] else '')
                                + (' ‡' if constant(r) else '') for r in rs], fontsize=6.3)
            ax.axvline(50, color=INK2, lw=0.8, ls='--')
            ax.set_xlim(0, 101); ax.set_title(LABELS[e], fontsize=9, color=INK, loc='left')
            ax.set_xlabel('Subject-wise balanced window accuracy (%)', fontsize=7.5, color=INK)
        h = [Line2D([], [], color=SERIES[0], marker='o', lw=1.6, label='other methods'),
             Line2D([], [], color=SERIES[1], marker='o', lw=1.6, label='starting method (SMVMD + mRMR + KNN)'),
             Line2D([], [], color=SERIES[0], marker='o', mfc='white', lw=1.6,
                    label='† uses unlabelled test-subject data')]
        fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        save(fig, FIG / 'all_methods_subjectwise'); plt.close(fig)
    # per-subject accuracy of the baseline and of the best method, from the saved predictions
    best = json.loads((GEN / 'best_models.json').read_text(encoding='utf-8')) if (GEN / 'best_models.json').exists() else {}
    panels = [(e, src, lab) for e in ('DS1_Rest', 'DS2') if e in best
              for src, lab in ((f'results/stage1_eval/{e}/knn_tableII__{sub_name(e)}.json', 'starting method'),
                               (best[e]['source'], 'best method'))]
    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), gridspec_kw={'width_ratios': [1, 2.6]})
        for ax, (e, src, lab) in zip(axes.T.flat, panels):
            z = np.load(ROOT / src.replace('.json', '.npz'), allow_pickle=True)
            y, p, g = z['y'].astype(int), z['y_pred'].astype(int), z['groups'].astype(str)
            subs = sorted(set(g), key=lambda s: (-y[g == s][0], s))
            acc = [100 * np.mean(p[g == s] == y[g == s]) for s in subs]
            cls = [y[g == s][0] for s in subs]
            style(ax)
            ax.bar(range(len(subs)), acc, 0.8, color=[SERIES[1] if c else SERIES[0] for c in cls], edgecolor=SURFACE, lw=0.3)
            ax.axhline(50, color=INK2, lw=0.8, ls='--')
            ax.set_ylim(0, 100); ax.set_xticks([])
            ax.set_xlabel(f'{"participant" if e.startswith("DS1") else "child"} (disorder first, then controls)',
                          fontsize=7, color=INK)
            ax.set_ylabel('windows correct (%)', fontsize=7.5, color=INK)
            ax.set_title(f'{LABELS[e]}: {lab}', fontsize=8, color=INK, loc='left')
        h = [plt.Rectangle((0, 0), 1, 1, color=SERIES[1], label='disorder'),
             plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label='control')]
        fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, FIG / 'per_subject_accuracy'); plt.close(fig)
    runs = [('True labels', R / 'stage2/DS1_Music/raw_shallow__subject_loso.json', SERIES[0]),
            ('Permuted labels 1', R / 'stage2_sanity/DS1_Music/raw_shallow_perm0__subject_loso.json', SERIES[3]),
            ('Permuted labels 2', R / 'stage2_sanity/DS1_Music/raw_shallow_perm1__subject_loso.json', SERIES[3])]
    runs = [(n, json.loads(p.read_text(encoding='utf-8')), c) for n, p, c in runs if p.exists()]
    if runs:
        fig, ax = plt.subplots(figsize=(5.2, 3.2)); style(ax)
        for i, (n, d, c) in enumerate(runs):
            v = 100 * d['metrics']['window']['accuracy']
            lo, hi = (100 * x for x in d['ci95']['window']['accuracy'])
            ax.bar(i, v, 0.6, color=c, edgecolor=SURFACE, lw=1)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=INK2, elinewidth=0.9, capsize=3)
            ax.text(i, 3, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color=SURFACE)
        ax.axhline(50, color=INK2, lw=0.8, ls='--')
        ax.set_xticks(range(len(runs))); ax.set_xticklabels([n for n, _, _ in runs], fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel('Window accuracy (%), LOSO', fontsize=8, color=INK)
        fig.tight_layout(); save(fig, FIG / 'permutation_test'); plt.close(fig)


def embedding_and_demo():
    s = load('stage1/results_summary.json')
    if s and 'embedding' in s:
        for e in ('DS1_Rest', 'DS1_Music', 'DS2'):
            v = s['embedding'].get(e)
            if v:
                mac(f'{TAG[e]}ProbeAcc', pct(v['subject_identification_accuracy_1nn'], 2))
                mac(f'{TAG[e]}ProbeChance', pct(v['subject_identification_chance'], 2))
                mac(f'{TAG[e]}SilClass', f"{v['silhouette_by_class']:.3f}")
                mac(f'{TAG[e]}SilSubj', f"{v['silhouette_by_subject']:.3f}")
    d = load('stage2/DS1_Rest/spectral_lr__subject_loso.json')     # the demo model's recipe
    if d:
        mac('DemoWinAcc', pct(d['metrics']['window']['accuracy']))
        mac('DemoWinAccCI', ci(d['ci95']['window']['accuracy']))
        mac('DemoWinAUC', f"{d['metrics']['window']['roc_auc']:.3f}")
        mac('DemoChildCorrect', f"{d['metrics']['child']['n_correct']}/{d['metrics']['child']['n']}")


def data_checks():
    d = load('data_checks/data_checks.json')
    if not d:
        return
    a = d['ds1_ages']
    mac('DSoneIddAgeMin', str(a['min'])); mac('DSoneIddAgeMax', str(a['max']))
    for c, t in (('rest', 'Rest'), ('music', 'Music')):
        b = d['ds1_separation'][c]['rel_beta']
        mac(f'DSone{t}RelBetaAUC', f"{b['auc_idd_higher']:.3f}")
        mac(f'DSone{t}RelBetaIdd', f"{b['idd_range'][0]:.3f}--{b['idd_range'][1]:.3f}")
        mac(f'DSone{t}RelBetaTdc', f"{b['tdc_range'][0]:.3f}--{b['tdc_range'][1]:.3f}")
    o = d['ds1_outliers']['recordings']
    if o:
        w = max(o, key=lambda r: r['sd_uv'])
        mac('DSoneOutlierFile', esc(w['file'].replace('_CD.mat', '')))
        mac('DSoneOutlierSD', f"{w['sd_uv']:.0f}"); mac('DSoneOutlierTimes', f"{w['times_median']:.0f}")
        mac('DSoneOutlierMax', f"{w['max_abs_uv']:.0f}"); mac('DSoneMedianSD', f"{d['ds1_outliers']['median_sd_uv']:.1f}")
    dup = d.get('ds1_duplicates')
    if dup:
        mac('DSoneNRecordings', str(dup['n_recordings']))
        mac('DSoneIdenticalPairs', str(len(dup['identical_pairs'])))
        mac('DSoneMaxPairCorr', f"{dup['max_abs_correlation']:.2f}")
    L = d['ds2_lengths']
    mac('DStwoLenAdhd', f"{L['adhd']['mean_windows']:.1f}"); mac('DStwoLenCtrl', f"{L['control']['mean_windows']:.1f}")
    mac('DStwoLenAUC', f"{L['auc_length_predicts_adhd']:.3f}")
    mac('DStwoLenAUCCI', '--'.join(f'{v:.3f}' for v in L['auc_ci95']))
    e = d['exact_ci_all_correct_ds1']
    mac('DSoneExactAllCorrectLow', pct(e['ci'][0]))


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for src in ['stage1_eval/figures/segment_vs_subject_gap.pdf', 'stage4/figures/channel_reduction_curve.pdf',
                'stage4/figures/channel_selection_topomap.pdf', 'stage5/figures/shap_spectral_xgb.pdf',
                'stage5/figures/eegnet_attribution.pdf']:
        p = R / src
        if p.exists():
            shutil.copy(p, FIG / p.name)
        else:
            missing.append(src)
    shutil.copy(ROOT / 'report/figures/IITI.png', FIG / 'IITI.png')


def plain(v):
    """LaTeX macro value -> Markdown text."""
    for a, b in (('--', '–'), ('\\_', '_'), ('\\%', '%'), ('\\&', '&'), ('\\#', '#'),
                 ('$^\\ast$', '*'), ('$^\\dagger$', '†'), ('$^\\ddagger$', '‡')):
        v = v.replace(a, b)
    return v


def panel_qa():
    """docs/panel_qa.md and docs/study_guide.md from their templates, numbers filled from the
    same macros."""
    import re
    vals = {}
    for x in macros:
        m = re.match(r'\\newcommand\{\\(\w+)\}\{(.*)\}$', x)
        if m:
            vals[m.group(1)] = m.group(2)
    for name in ('panel_qa', 'study_guide'):
        src = ROOT / f'docs/{name}.template.md'
        if not src.exists():
            continue
        t = src.read_text(encoding='utf-8')
        out = re.sub(r'\{\{(\w+)\}\}', lambda m: plain(vals.get(m.group(1), '??')), t)
        (ROOT / f'docs/{name}.md').write_text(out, encoding='utf-8')
        n = out.count('??') - t.count('??')     # the template's own comment mentions ??
        if n:
            print(f"docs/{name}.md: {n} values not yet computed (??)")


def overleaf_zip():
    """report/final/BTP_final_overleaf.zip: upload to Overleaf and compile BTP_final.tex."""
    import zipfile
    z = OUT / 'BTP_final_overleaf.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p != z and p.suffix in ('.tex', '.pdf', '.png'):
                f.write(p, p.relative_to(OUT))
    print(f'Overleaf bundle: {z.relative_to(ROOT)}')


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    gap(); models(); study_tables(); ladder(); cross(); explain(); embedding_and_demo(); dataset_stats()
    data_checks(); figures(); feature_figures(); study_figures()
    (GEN / 'final_numbers.tex').write_text('% generated by report/build_final.py from results/ -- do not edit\n'
                                           + '\n'.join(macros) + '\n', encoding='utf-8')
    panel_qa(); overleaf_zip()
    print(f'{len(macros)} macros, tables in {GEN}')
    if missing:
        print('MISSING (macros/tables incomplete):'); [print('  ', m) for m in missing]


if __name__ == '__main__':
    main()
