# Stage 0 report: baseline verification

Date: 4 Nov 2026. Machine: Windows 11, 12 cores, 8 GB RAM, no GPU, Python 3.13.3.

**Result: the README numbers reproduce exactly (bit-for-bit), not just within
noise.** No bug was found that changes a reported number. `ds1_rest`, `ds1_music`,
`ds1_rest_music` and the `eval`/`raw` parts of `stage1_analysis` were not run
as scripts because they exceed the 30-minute budget (§4). Their headline
numbers were recomputed directly instead (§2).

All verification ran in an isolated copy of the repository so that the
existing result files, logs and the uncommitted report work were not touched.
The repository's `results/` tree is unchanged by Stage 0.

---

## 1. Dependencies

Everything the code imports was already installed (numpy 2.4.5, scipy 1.17.1,
scikit-learn 1.8.0, pandas 3.0.3, mne 1.13.2, PyWavelets 1.10.0,
matplotlib 3.11.1, scikit-optimize 0.10.2, torch 2.13.0 CPU). Nothing new was
installed.

`requirements.txt` does not match the code:

- **Missing:** `reportlab` (imported by `report/build_project_record.py`).
- **Listed but never imported:** `mrmr-selection`, `bayesian-optimization`,
  `jupyter`, `ipykernel`, `ipywidgets`, `pdfplumber` (dev only).
- `run_guide.md` Step 0 asks for `pip install mne seaborn h5py openpyxl`
  separately, but these are already in `requirements.txt`.

## 2. What ran

| Check | What it proves | Result | Runtime |
|---|---|---|---|
| Feature-cache smoke test: recompute SMVMD + features from raw EEG for 7 windows (DS-1 rest: IDD_001 w0, w20, TDC_007 w45; DS-2: v107 w0, w30, v10p w0, w21) and compare to `results/features/*.npz` | The cached matrices are what the **current** code produces (so the earlier PCC fix is in the cache) | Max relative difference **0.0** on every window; mode counts identical | 34 s |
| DS-1 rest+music cache vs stacked music + rest caches (IDD_001, IDD_002) | Rest+music is the same per-window features, music then rest | Max abs difference 0.0 | < 1 s |
| Window counts from raw | Segmentation | DS-1: 46 windows per recording; DS-2 v107: 60, v10p: 43, matching the cache | — |
| Headline protocols re-run with the `stage1_analysis` pipeline, splitters and scoring (table below) | README numbers | All identical | 21 min total |
| `python -m experiments.ds2_adhd` (full entry point, cached features) | Entry point runs end to end | see §3 | see §3 |
| `stage1_analysis --part reproduction / modes / embed / features` | Cheap analysis parts | see §3 | see §3 |

### Headline numbers: recomputed vs `results/stage1/results_summary.json`

| Setting | Protocol | Window acc. (recomputed) | Window acc. (stored) | Children correct (recomputed / stored) | Runtime |
|---|---|---|---|---|---|
| DS-1 rest | Subject-wise (LOSO) | 62.73% | 62.73% | 10/14 / 10/14 | 234 s |
| DS-1 music | Subject-wise (LOSO) | 43.01% | 43.01% | 8/14 / 8/14 | 278 s |
| DS-1 rest+music | Subject-wise (LOSO) | 62.34% | 62.34% | 9/14 / 9/14 | 616 s |
| DS-2 ADHD | Subject-wise (GroupKFold 10) | 57.54% | 57.54% | 74/121 / 74/121 | 7 s |
| DS-1 rest | Segment-wise 10-fold | 99.69% | 99.69% | 14/14 / 14/14 | 106 s |
| DS-2 ADHD | Segment-wise 10-fold | 85.81% | 85.81% | 121/121 / 121/121 | 16 s |
| DS-1 rest | Child-ID 1-NN probe | 98.91% | 98.91% | — | < 10 s |

Values were compared at full floating-point precision, not only at two
decimals. Source: `verify_headline.json` from the verification run (scratch,
not committed). DS-2 results are subject to the medication confound: the ADHD
children were taking methylphenidate and the controls were not.

## 3. Entry-point runs

Run in the isolated copy, with the cached features. Outputs were diffed
key by key against a backup of the repository's `results/`.

| Command | Exit | Wall time | Diff against stored results |
|---|---|---|---|
| `python -m experiments.ds2_adhd` (with phase 1–5 sanity checks) | 0 | 929 s (15.5 min) | 17 of 107 values in `metrics/DS2.json` differ. **All are timings, `run_at` or the OS build string.** Every metric is identical: paper-protocol accuracy 85.81%, subject-independent 59.21%, classifier comparison DT 70.04%, KNN 85.81%, SVM 65.29%, Ensemble 80.69%. |
| `stage1_analysis --part reproduction` | 0 | 13 s | Only `_updated`, the source timestamp and the classifier timing fields differ |
| `stage1_analysis --part modes` | 0 | 8 s | identical |
| `stage1_analysis --part embed` | 0 | 184 s | identical (silhouettes, child-ID probe on all four settings) |
| `stage1_analysis --part features` | 0 | 15 s | identical |

One side effect: a cached run **overwrites** `metrics/DS2.json` and drops the
original preprocessing, SMVMD and feature timings, because those stages did
not run (`timings_sec.features` 3473 s → absent). If the repository's
`ds*_*.py` scripts are re-run, the record of how long feature extraction took
is lost. Stage 1 should store feature timings inside the feature cache.

## 4. Not run (over the 30-minute budget), with estimates

The README says cached re-runs "take seconds". **That is not true.** The cache
skips only preprocessing, SMVMD and feature extraction. mRMR (a 126×126 or
171×171 mutual-information matrix per fold) and the CV loops run every time.
These estimates come from the stage timings stored in `results/metrics/*.json`
and `results/logs/stage1_analysis.log`:

| Command | Cached runtime (estimate) |
|---|---|
| `python -m experiments.ds1_rest` | ≈ 28 min (mRMR curve 4 min, paper CV 12 min, subject CV 12 min) |
| `python -m experiments.ds1_music` | ≈ 20 min |
| `python -m experiments.ds1_rest_music` | ≈ 65 min |
| `python -m experiments.stage1_analysis --part eval` | ≈ 64 min |
| `python -m experiments.stage1_analysis --part raw` (features cached) | ≈ 32 min |
| **Total** | **≈ 3.5 h** |

The protocol numbers these scripts report were recomputed directly (§2), and
the features were verified against raw EEG. Running the scripts again would
mostly re-check the paper-protocol path in `src/knn_model.py` and the mRMR
curve plots. They can run overnight if you want that.

## 5. Discrepancies and code-quality issues

None of these changes a reported number. They are listed roughly by
importance. **None has been fixed**; items 1–2 need your decision.

1. **DS-2 is wavelet-denoised twice.** `preprocess_ds2` calls
   `wavelet_filter_ocular` and then `sure_wavelet_denoise`
   (`src/preprocessing.py`). The two functions are the same operation:
   coif3, level 6, SURE soft threshold on every detail level, approximation
   kept. The paper describes one Coiflet-3 wavelet filtering step with SURE up
   to level 6. The docstring of `wavelet_filter_ocular` says it "zeroes"
   low-frequency coefficients, but the code does not. Soft thresholding
   removes *small* coefficients, so neither step specifically removes
   (large, slow) ocular artefacts. This is a preprocessing deviation. Fixing
   it would change the DS-2 features and every DS-2 number, and would need a
   full DS-2 recompute (about 75 min of preprocessing + features). **Decision
   needed:** keep it as a documented deviation, or fix it (as a separate,
   labelled variant).
2. **Majority-vote ties go to the control class.** `score()` in
   `experiments/stage1_analysis.py` sets `vote = frac1 > 0.5`. DS-2 has 3
   ties in the subject-wise run. This is documented in the code but biases
   the child-level confusion matrix. Stage 1 should break ties with the mean
   predicted probability and report the tie count.
3. **The mRMR accuracy curve (paper Fig. 5) leaks and ignores
   standardisation.** `run_ds1_experiment` ranks features on all windows, then
   `select_features_by_accuracy` runs segment-wise CV with
   `build_knn_classifier`, which never applies the `standardize` flag. This
   affects only the curve plot and the `SEG10_GLOBAL` protocol, which is
   labelled as leaky on purpose, not the headline numbers.
4. **DS-2 subject-wise split is `GroupKFold(10)` without stratification or
   shuffling.** There is no leakage, but class balance per fold is not
   controlled and the split cannot be re-seeded. Stage 1 will use
   `StratifiedGroupKFold` with a seed.
5. **0%-overlap mask assumes exactly 46 windows per DS-1 recording**
   (`window_position`, `pos % 46`). This holds now, but nothing asserts it.
6. **Hard-coded seeds.** `mutual_info_classif(random_state=42)` in
   `src/mrmr.py` ignores the `random_seed` argument, and `SEED = 42` in
   `stage1_analysis.py` does not read `config.yaml`.
7. **No tests, no `.gitignore`.** `__pycache__/` folders show up as
   untracked. Both datasets (222 files, including the duplicate unzipped copy
   in `data/raw/ds2/`) and the feature caches are committed to git. Check that
   redistribution is allowed if the repo is public: both datasets have their
   own licences.
8. **Slow SURE threshold.** `_sure_threshold` is O(n²) per wavelet level. It
   is the main reason DS-2 preprocessing takes about 12 min. A sorted-cumsum
   version is O(n log n) and gives identical output.
9. **Documentation drift.** README/run_guide "re-runs take seconds" (see §4).
   `docs/reproducibility_notes.md` §2.1 still contains a draft note ("Wait -
   the paper equation (4)...") and an older stopping-criterion description
   that contradicts §2.2. `config.yaml` says
   `subject_independent.n_splits: 10`, but DS-1 actually uses LOSO.
10. **Unused or duplicated code.** `src/knn_model.py` has
    `bayesian_optimize_knn` (never called) and its own subject-independent
    evaluator (`StratifiedGroupKFold(10)` on 14 DS-1 children), whose numbers
    end up in `results/metrics/*.json` as `subject_independent` (for example
    DS-1 rest 61.96%) alongside the LOSO numbers in `results_summary.json`
    (62.73%). Two different "subject-wise" numbers exist on disk. The README
    uses the LOSO one. Stage 1 should make that the only one.
11. Observation, not an issue: the raw-channel baseline (same 9 features, no
    SMVMD) is at least as good subject-wise as SMVMD on every setting (for
    example DS-1 rest 66.15% vs 62.73%; `results_summary.json`, `raw_baseline`).
    This is already in the summary and matters for the Stage 3 story.

## 6. Uncommitted work found in the repo

At the start of Stage 0 these were modified or untracked and are **yours**.
They were not touched or committed: `experiments/stage1_analysis.py`
(`part_features`), `experiments/classifier_sweep.py`, `report/*`,
`results/stage1/*` (feature-composition outputs), `results/logs/stage1_analysis.log`.
Please commit them, or tell me to, before Stage 1, so the Stage 1 commit
contains only Stage 1 changes.
