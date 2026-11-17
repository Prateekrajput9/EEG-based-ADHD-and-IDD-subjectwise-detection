"""Live EEG pipeline: source -> streaming preprocessor -> windowed inference.

    sources.py     ReplaySource (recorded DS-1 file at real-time rate), LSLSource (headset)
    preprocess.py  ring buffer, optional causal filters, 5 s / 2.5 s windows, resampling,
                   channel mapping, signal-quality flags
    inference.py   saved model bundle -> P(IDD) per window, smoothing, latency
    app.py         Streamlit dashboard (streamlit run src/live/app.py)

Research demo only; not a diagnostic tool.
"""
DISCLAIMER = ('Research demonstration only - NOT a medical device and NOT a diagnosis. '
              'Trained on 14 adults (7 with IDD, 7 typically developing controls) recorded with one '
              'headset; its accuracy on new people is far below published figures and very uncertain, '
              'and an output for anyone recorded on another device or session means nothing. '
              'Never use it to make decisions about a person.')
