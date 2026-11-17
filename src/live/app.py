"""Live EEG dashboard (research demo).

    streamlit run src/live/app.py                       # replay (default, offline)
    streamlit run src/live/app.py -- --source lsl       # headset via LSL

Shows: live multichannel EEG, relative band powers, windowed P(IDD) with
smoothing, per-channel signal quality, latency, and the segment-wise vs
subject-wise accuracy gap read from the saved results.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.live import DISCLAIMER
from src.live.inference import LiveModel, band_powers
from src.live.preprocess import StreamingPreprocessor, quality
from src.live.sources import LSLSource, ReplaySource
from src.utils import load_config

BLUE, ORANGE, INK2, GRID = '#2a78d6', '#eb6834', '#52514e', '#e4e3df'


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['replay', 'lsl'], default=None)
    ap.add_argument('--config', default=str(ROOT / 'configs/live.yaml'))
    return ap.parse_args(sys.argv[1:])


def init_state(live, source_kind, subject, condition):
    cfg = load_config(str(ROOT / 'configs/config.yaml'))
    if source_kind == 'replay':
        src = ReplaySource.from_ds1(cfg, subject, condition, speed=live['replay']['speed'],
                                    loop=live['replay']['loop'])
        filt = {}
    else:
        L = live['lsl']
        src = LSLSource(L['stream_type'], L['stream_name'], L['channels'], L['resolve_timeout_sec'])
        filt = {'bandpass_hz': L['bandpass_hz'], 'notch_hz': L['notch_hz'], 'remove_dc': L['remove_dc']}
    src.start()
    st.session_state.src = src
    st.session_state.pre = StreamingPreprocessor(src.srate, len(src.channels), live['window_sec'],
                                                 live['hop_sec'], live['fs'], **filt)
    st.session_state.model = LiveModel(str(ROOT / live['model']), live['smoothing_windows'])
    st.session_state.log = []
    st.session_state.last = None
    st.session_state.key = (source_kind, subject, condition)


def eeg_figure(x, channels, srate):
    t = np.arange(x.shape[1]) / srate
    fig = go.Figure()
    sd = np.median(x.std(1)) * 6 + 1e-9
    for i, c in enumerate(channels):
        fig.add_trace(go.Scattergl(x=t, y=x[i] - x[i].mean() - i * sd, mode='lines', name=c,
                                   line=dict(width=1, color=BLUE), hovertemplate=f'{c}: %{{y:.1f}}<extra></extra>'))
    fig.update_layout(height=430, margin=dict(l=50, r=10, t=10, b=30), showlegend=False,
                      plot_bgcolor='#fcfcfb',
                      yaxis=dict(tickvals=[-i * sd for i in range(len(channels))], ticktext=channels,
                                 showgrid=False),
                      xaxis=dict(title='seconds (last 10 s)', gridcolor=GRID))
    return fig


def prob_figure(log):
    fig = go.Figure()
    if log:
        t = [r['t'] for r in log]
        fig.add_trace(go.Scatter(x=t, y=[r['p_idd'] for r in log], name='per window', mode='markers',
                                 marker=dict(size=8, color=BLUE)))
        fig.add_trace(go.Scatter(x=t, y=[r['p_idd_smoothed'] for r in log], name='smoothed',
                                 mode='lines', line=dict(width=2, color=ORANGE)))
    fig.add_hline(y=0.5, line=dict(color=INK2, dash='dash', width=1))
    fig.update_layout(height=260, margin=dict(l=50, r=10, t=10, b=30), plot_bgcolor='#fcfcfb',
                      yaxis=dict(range=[0, 1], title='P(IDD)', gridcolor=GRID),
                      xaxis=dict(title='seconds since start', gridcolor=GRID),
                      legend=dict(orientation='h', y=1.12))
    return fig


def gap_panel():
    root = ROOT / 'results/stage1_eval'
    rows = []
    for exp, lab in (('DS1_Rest', 'DS-1 rest'), ('DS1_Music', 'DS-1 music'),
                     ('DS1_RestMusic', 'DS-1 rest+music'), ('DS2', 'DS-2 ADHD*')):
        sub = 'subject_loso' if exp.startswith('DS1') else 'subject_sgkf'
        a, b = root / exp / 'knn_tableII__segment10.json', root / exp / f'knn_tableII__{sub}.json'
        if a.exists() and b.exists():
            A, B = json.loads(a.read_text()), json.loads(b.read_text())
            rows.append((lab, A['metrics']['window']['accuracy'], B['metrics']['window']['accuracy'],
                         B['ci95']['window']['accuracy']))
    if not rows:
        st.info('Run `python -m experiments.stage1_eval` to create the gap results.'); return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1] * 100 for r in rows], name='Segment-wise split (leaky)',
                         marker_color=BLUE, hovertemplate='%{y:.1f}%<extra>segment-wise</extra>'))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[2] * 100 for r in rows], name='Subject-wise (new people)',
                         marker_color=ORANGE,
                         error_y=dict(type='data', symmetric=False,
                                      array=[(r[3][1] - r[2]) * 100 for r in rows],
                                      arrayminus=[(r[2] - r[3][0]) * 100 for r in rows], color=INK2),
                         hovertemplate='%{y:.1f}%<extra>subject-wise</extra>'))
    fig.add_hline(y=50, line=dict(color=INK2, dash='dash', width=1))
    fig.update_layout(barmode='group', height=330, yaxis=dict(range=[0, 105], title='window accuracy (%)'),
                      plot_bgcolor='#fcfcfb', margin=dict(l=50, r=10, t=30, b=30),
                      legend=dict(orientation='h', y=1.15))
    st.plotly_chart(fig, width='stretch')
    st.caption('Same model (SMVMD + mRMR + KNN, paper settings) under two splits; error bars: 95% bootstrap '
               'CI over subjects. DS-1 participants are adults. *DS-2 ADHD children were on '
               'methylphenidate, controls were not.')
    fig2 = ROOT / 'results/stage1_eval/figures/segment_vs_subject_gap.png'
    if fig2.exists():
        with st.expander('Static figure (report version)'):
            st.image(str(fig2))


def main():
    a = args()
    live = yaml.safe_load(Path(a.config).read_text())
    st.set_page_config(page_title='EEG research demo', layout='wide')
    st.error(DISCLAIMER, icon='⚠️')
    with st.sidebar:
        st.header('Source')
        kind = st.radio('EEG source', ['replay', 'lsl'],
                        index=0 if (a.source or live['source']) == 'replay' else 1,
                        format_func=lambda k: 'Replay recorded file' if k == 'replay' else 'Headset (LSL)')
        subs = [f'IDD_{i:03d}' for i in range(1, 8)] + [f'TDC_{i:03d}' for i in range(1, 8)]
        subject = st.selectbox('Replay participant', subs, index=subs.index(live['replay']['subject']),
                               disabled=kind != 'replay')
        condition = st.selectbox('Condition', ['rest', 'music'], disabled=kind != 'replay')
        st.caption('Replayed participants were part of training data for the demo model, so their '
                   'predictions are NOT a test. Honest new-person numbers are below.')
        if st.button('Restart') or st.session_state.get('key') != (kind, subject, condition):
            try:
                init_state(live, kind, subject, condition)
            except Exception as e:
                st.session_state.pop('src', None); st.warning(str(e))
        b = st.session_state.get('model').bundle if st.session_state.get('model') else None
        if b and b.get('honest'):
            h = b['honest']
            st.metric('Recipe accuracy on NEW people (window)', f"{h['window']['accuracy']*100:.1f}%",
                      help=f"95% CI {h['ci95']['window']['accuracy'][0]*100:.1f}-"
                           f"{h['ci95']['window']['accuracy'][1]*100:.1f}%; source {h['source']}")
            st.metric('Participants classified correctly (LOSO)', f"{h['child']['n_correct']}/{h['child']['n']}")

    tab_live, tab_gap = st.tabs(['Live', 'Why new people are hard (accuracy gap)'])
    with tab_gap:
        gap_panel()
    with tab_live:
        if 'src' not in st.session_state:
            st.stop()
        live_fragment(live)


@st.fragment(run_every=0.5)
def live_fragment(live):
    s = st.session_state
    t_now = time.perf_counter()
    chunk = s.src.read()
    for w in s.pre.push(chunk):
        r = s.model.predict(w, t_window_end=t_now)
        q = quality(w, live['quality']['flat_sd_uv'], live['quality']['max_abs_uv'])
        r.update({'t': s.pre.next_start / s.pre.srate, 'bands': band_powers(w), 'quality': q})
        s.log.append(r); s.last = r
    c1, c2 = st.columns([3, 2])
    with c1:
        x = s.pre.recent(10)
        if x.shape[1] > 10:
            st.plotly_chart(eeg_figure(x, s.src.channels, s.pre.srate), width='stretch')
        else:
            st.info('Waiting for data...')
    with c2:
        if s.last:
            r = s.last
            st.metric('P(IDD), smoothed over last windows', f"{r['p_idd_smoothed']:.2f}",
                      help='Model output, not a diagnosis. 0.5 = undecided.')
            lat = r['latency_ms']
            st.caption(f"window compute {r['compute_ms']:.0f} ms · end-of-window to result "
                       f"{lat:.0f} ms · windows so far {len(s.log)}")
            bp = r['bands']
            fig = go.Figure(go.Bar(x=list(bp), y=[v * 100 for v in bp.values()], marker_color=BLUE,
                                   hovertemplate='%{x}: %{y:.1f}%<extra></extra>'))
            fig.update_layout(height=200, margin=dict(l=40, r=10, t=10, b=30), plot_bgcolor='#fcfcfb',
                              yaxis=dict(title='relative power (%)', gridcolor=GRID))
            st.plotly_chart(fig, width='stretch')
            q = r['quality']
            bad = [f'{c} ({f})' for c, f in zip(s.src.channels, q['flags']) if f != 'ok']
            (st.success if not bad else st.warning)(
                f"Signal quality: {q['fraction_ok']*100:.0f}% of channels OK" + (f" — check {', '.join(bad)}" if bad else ''))
        else:
            st.info('First prediction after 5 s of data.')
    st.plotly_chart(prob_figure(s.log[-60:]), width='stretch')


if __name__ == '__main__':
    main()
