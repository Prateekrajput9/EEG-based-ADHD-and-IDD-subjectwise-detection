"""Comparison tables built only from saved result JSON files."""
import csv
import json
from pathlib import Path
from typing import Dict, List

ORDER = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']


def _pct(v):
    return '—' if v is None else f'{100 * v:.1f}'


def _ci(c):
    return '—' if not c or c[0] is None else f'{100 * c[0]:.0f}–{100 * c[1]:.0f}'


def collect(roots: List[str], include_leaky: bool = False) -> List[Dict]:
    rows = []
    for root in roots:
        for p in sorted(Path(root).glob('*/*.json')):
            d = json.loads(p.read_text(encoding='utf-8'))
            if 'metrics' not in d:
                continue
            if d['protocol'].get('leaky') and not include_leaky:
                continue
            w, c, ci = d['metrics']['window'], d['metrics']['child'], d['ci95']
            rows.append({
                'experiment': d['experiment'], 'features': d['model'].get('features', 'SMVMD'),
                'model': d['model']['name'], 'protocol': d['protocol']['name'],
                'transductive': bool(d['model'].get('uses_unlabelled_test_child_data', False)),
                'window_acc': w['accuracy'], 'window_acc_ci': ci['window']['accuracy'],
                'window_bal_acc': w['balanced_accuracy'], 'window_bal_acc_ci': ci['window']['balanced_accuracy'],
                'window_auc': w['roc_auc'], 'window_auc_ci': ci['window']['roc_auc'],
                'window_f1': w['f1'],
                'child_acc': c['accuracy'], 'child_acc_ci': ci['child']['accuracy'],
                'child_bal_acc': c['balanced_accuracy'], 'child_auc': c['roc_auc'],
                'child_auc_ci': ci['child']['roc_auc'],
                'child_correct': c['n_correct'], 'n_children': c['n'], 'vote_ties': c['n_vote_ties'],
                'runtime_sec': d.get('runtime_sec'), 'source': str(p).replace('\\', '/'),
            })
    rows.sort(key=lambda r: (ORDER.index(r['experiment']) if r['experiment'] in ORDER else 9,
                             -(r['window_bal_acc'] or 0)))
    return rows


def write(rows: List[Dict], base: Path, title: str, note: str = '') -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(base.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    base.with_suffix('.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    L = [f'# {title}', '', 'Generated from the result JSON files listed in the CSV `source` column. '
         'Percentages; 95% CIs are percentile bootstrap over children (stratified by class).', '']
    if note:
        L += [note, '']
    cur = None
    for r in rows:
        if r['experiment'] != cur:
            cur = r['experiment']
            L += ['', f'## {cur}', '',
                  '| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | '
                  'Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        flag = ' (†)' if r['transductive'] else ''
        auc = lambda v: '—' if v is None else f'{v:.3f}'
        aci = lambda c: '—' if not c or c[0] is None else f'{c[0]:.2f}–{c[1]:.2f}'
        L.append(f"| {r['features']} | {r['model']}{flag} | {r['protocol']} | "
                 f"{_pct(r['window_acc'])} [{_ci(r['window_acc_ci'])}] | "
                 f"{_pct(r['window_bal_acc'])} [{_ci(r['window_bal_acc_ci'])}] | "
                 f"{auc(r['window_auc'])} [{aci(r['window_auc_ci'])}] | "
                 f"{_pct(r['child_acc'])} [{_ci(r['child_acc_ci'])}] | "
                 f"{r['child_correct']}/{r['n_children']} | {auc(r['child_auc'])} [{aci(r['child_auc_ci'])}] | "
                 f"{r['vote_ties']} | {r['runtime_sec']:.0f} |")
    L += ['', '(†) uses unlabelled data of the test child at test time (transductive); '
          'not comparable to strictly inductive rows.',
          '', 'DS-2: ADHD children were taking methylphenidate and the controls were not; '
          'ADHD cannot be separated from its treatment in this dataset.']
    base.with_suffix('.md').write_text('\n'.join(L), encoding='utf-8')
