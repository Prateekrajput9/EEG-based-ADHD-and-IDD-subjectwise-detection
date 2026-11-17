# Study guide: every topic and method in the project

<!-- docs/study_guide.md is generated from docs/study_guide.template.md by
     `python report/build_final.py`. Edit the template, not the output.
     Double-brace placeholders are filled from results/; ?? means not computed yet. -->

> Research prototype, not a diagnostic tool. DS-2 caveat: the ADHD children were on
> methylphenidate (Ritalin) and the controls were not. DS-1 caveat: the IDD participants are
> adults (26–31 years).

**How to use this guide.** Each method has the same five parts: what it is, how it works, why
it is in the project, what you found, and the question a panel is likely to ask. Every number
is filled in from the saved results, so it always matches the report. For a topic to count as
learnt, you should be able to explain it out loud in about a minute without notes.

Priority marks: ★★★ expect to be asked · ★★ asked if it is on a slide · ★ background.

**Contents**
1. The project in one page
2. Background: EEG, ADHD, IDD
3. The data
4. Signal processing basics
5. The paper's pipeline (SMVMD + mRMR + KNN)
6. The evaluation protocol (your main contribution)
7. Statistics used in evaluation
8. The other feature sets
9. The other models
10. Ablations and extra experiments
11. Explaining the models
12. Checks for leaks and confounds
13. The live demo
14. Key numbers to memorise
15. Self-test

---

## 1. The project in one page ★★★

**Problem statement.** EEG-based detection of multiple neurodevelopmental disorders (IDD and
ADHD).

**Story in five steps.**
1. A 2025 IEEE paper (Chandela et al.) reports 100% (IDD) and
   99.17% (ADHD) accuracy with SMVMD + mRMR + KNN.
2. I reproduced the pipeline: 99.69% on DS-1 rest and 85.81% on
   DS-2, under the paper's window-wise split.
3. Tested the way it would be used, on people the model has never seen, it gives
   62.73% and 59.30%.
4. The reason: the features identify the *person* (98.91% person
   identification, chance 7.14%), and the window-wise split rewards that.
5. I compared 24 methods per setting under the strict protocol, ran
   ablations, confound checks and explanations, and built a real-time demo.

**What is yours:** the reproduction, the leakage-safe evaluation framework, the measured gap
and its explanation, the 24-method benchmark, the ablation showing SMVMD is not needed, the
confound findings, and the live demo. **What is not yours:** the individual methods (SMVMD,
KNN, Riemannian geometry, EEGNet and so on) and the datasets.

---

## 2. Background

### EEG ★★★
- **What:** voltage on the scalp produced by the summed activity of many neurons, measured in
  microvolts (µV).
- **Electrodes:** placed by the international 10–20 system. Letters give the region (F
  frontal, T temporal, P parietal, O occipital, C central); odd numbers are left, even are
  right, z is the midline.
- **Frequency bands:** delta 1–4 Hz (deep sleep), theta 4–8 Hz (drowsiness, memory), alpha
  8–13 Hz (relaxed, eyes closed), beta 13–30 Hz (active thinking, also muscle tension), gamma
  above 30 Hz (strongly contaminated by muscle).
- **Artefacts:** eye blinks and movements (large, slow, frontal), muscle (broadband, high
  frequency, temporal and frontal), mains hum (50 Hz in India), poor electrode contact.

### ADHD ★★
- Attention-deficit/hyperactivity disorder: inattention, hyperactivity, impulsivity.
- Diagnosed clinically from behaviour (DSM criteria); about 5–7% of children.
- Methylphenidate (Ritalin) is a stimulant treatment and changes the EEG.
- The theta/beta ratio was proposed as an EEG marker of ADHD; later studies did not confirm
  it reliably.

### IDD ★★
- Intellectual developmental disorder: IQ below about 70 plus difficulties with everyday
  adaptive skills; about 1–2% of people.

**Likely question:** "Why use EEG at all?" It is cheap, portable and non-invasive, and could
give an objective aid to a diagnosis that is currently based on behaviour and judgement.

---

## 3. The data ★★★

| | DS-1 (IDD) | DS-2 (ADHD) |
|---|---|---|
| Participants | 7 IDD + 7 typically developing controls | 61 ADHD + 60 controls |
| Age | IDD 26–31 years; controls not given | 7–12 years |
| Device | EMOTIV EPOC+, 14 channels, consumer headset | clinical EEG, 19 channels |
| Sampling rate | 128 Hz | 128 Hz |
| Task | 2 min rest, 2 min music | visual attention task (counting cartoon characters) |
| Settings in results | rest, music, rest+music | one |
| Windows | 644 per condition (46 per recording) | 6588 |

**Things to know about the data.**
- DS-1 is tiny: one participant is 7 percentage points of participant-level accuracy.
- DS-1 is adults, although the base paper describes its study as detection in children.
- DS-2 compares medicated ADHD children with unmedicated controls.
- In DS-2 the task ran until the child finished; ADHD children took longer
  (60.3 windows on average against 48.5).
- DS-2 labels come from which ZIP file a subject is in.

**Windowing.** Each recording is cut into 5-second windows (640 samples) with 50% overlap, so
consecutive windows share half their samples. Every window is labelled with its person's
diagnosis.

---

## 4. Signal processing basics

### Sampling and the Nyquist frequency ★★★
At 128 Hz the highest frequency that can be represented is 64 Hz (half the sampling rate).
Higher frequencies would fold back as false lower ones (aliasing).

### Fourier transform and power spectrum ★★★
- **What:** any signal can be written as a sum of sine waves; the Fourier transform gives the
  strength of each frequency.
- **Power spectral density (PSD):** power per frequency.
- **Welch's method:** split the window into overlapping segments, compute each spectrum and
  average them, which gives a smoother, more reliable estimate. Your code uses 256-sample
  segments, so the frequency resolution is 128/256 = 0.5 Hz.
- **Band power:** the area under the PSD inside a band. **Relative band power** divides by the
  total power, so it does not depend on the overall signal size.

### Filters ★★★
- **Band-pass:** keeps a frequency range (DS-2: 0.5–60 Hz; live demo: 1–30 Hz).
- **Notch:** removes one narrow frequency, here 50 Hz mains hum.
- **Butterworth:** a standard filter design with a flat passband; its order sets how sharply
  it cuts off.
- **Zero-phase vs causal:** offline code filters forwards and backwards (no time shift, but it
  needs future samples). The live demo must filter causally (only past samples), which delays
  the signal slightly.

### Wavelet denoising ★★
- **Wavelet transform:** splits the signal into frequency bands with time information kept,
  using short waves of different scales. At 128 Hz, level 1 holds 32–64 Hz, level 2 16–32 Hz,
  and so on down to level 6 (1–2 Hz).
- **DS-2 preprocessing:** Coiflet-3 wavelet, level 6, with SURE thresholding: small
  coefficients (mostly noise) are shrunk towards zero, using a threshold that minimises an
  estimate of the error.

### Hilbert transform and analytic signal ★
Turns a real signal into a complex one with only positive frequencies, which gives the
instantaneous amplitude and phase. Used inside VMD and for wPLI.

---

## 5. The paper's pipeline

```
EEG → preprocessing → 5 s windows → SMVMD → 9 features per mode per channel
    → energy-weighted integration → mRMR → KNN
```

### 5.1 VMD, MVMD and SMVMD ★★★
- **VMD (variational mode decomposition):** splits a signal into K "modes", each a narrow-band
  oscillation around its own centre frequency, chosen so that the modes add up to the signal
  and their total bandwidth is as small as possible.
- **How VMD is solved:** an optimisation with a constraint, solved by alternating three simple
  updates (ADMM):
  1. update each mode with a band-pass filter centred on its frequency:
     `û_k(ω) = [x̂(ω) − Σ other modes + λ̂/2] / [1 + 2α(ω − ω_k)²]`
  2. move each centre frequency to the centre of gravity of its mode's spectrum;
  3. update the Lagrange multiplier λ (with step τ).
- **Parameters:** α is the bandwidth penalty (larger α gives narrower modes): 2000 for DS-1 and
  1000 for DS-2. τ = 0 means the modes need not rebuild the signal exactly, which tolerates
  noise. The tolerance sets when to stop iterating.
- **MVMD (multivariate VMD):** decomposes all channels together, with one shared centre
  frequency per mode, so modes are aligned across channels.
- **SMVMD (successive MVMD):** extracts modes one at a time from what is left of the signal,
  keeping each new mode away from earlier ones, and stops when little energy remains (your
  code: under 1% residual energy, or 20 modes at most). So K is chosen automatically and
  differs between windows.
- **Compared with EMD:** EMD (empirical mode decomposition) is a heuristic and suffers from
  mode mixing; VMD has a mathematical formulation but needs K; SMVMD removes the need for K.
- **Your result:** SMVMD did not help under the strict protocol (see the ablation ladder,
  section 10.1).
- **Likely question:** "What does α do?" or "Why does K vary?"

### 5.2 The nine features ★★★
Computed on every mode of every channel:

| Feature | What it measures |
|---|---|
| STD, VAR, RMS | size of the signal (standard deviation, variance, root mean square) |
| IQR | spread of the middle 50% of values, robust to outliers |
| ASSR | sum of square roots of absolute values: size, less driven by peaks |
| Average power (AP) | mean of the windowed power spectrum |
| Maximum fractal length (MFL) | log of the waveform's total "length": complexity and fast activity |
| Information potential (IP) | from the estimated amplitude distribution; low IP means a spread-out (high-entropy) distribution |
| Parametric-centred correntropy (PCC) | a nonlinear version of how similar the signal is to itself one sample later |

IP and PCC use a Gaussian kernel whose width is set by Silverman's rule
(σ = 1.06 · std · N^(−1/5)).

### 5.3 Energy-weighted integration (paper's Eq. 7) ★★★
- **Problem:** SMVMD gives a different number of modes per window, so the feature count would
  vary, and classifiers need a fixed length.
- **Solution:** for each channel and feature, take a weighted average over modes, where each
  mode's weight is its energy (sum of squared samples).
- **Result:** 9 numbers per channel: 14 × 9 = 126 for DS-1 and 19 × 9 = 171 for DS-2.

### 5.4 mRMR feature selection ★★★
- **Mutual information (MI):** how much knowing one variable reduces uncertainty about
  another; zero only if they are independent, and it captures nonlinear relationships.
- **mRMR (minimum redundancy, maximum relevance):** picks features one at a time with the
  highest `MI(feature; label) − average MI(feature; already chosen features)`.
- **In your code:** MI is estimated with k-nearest-neighbour estimators (scikit-learn), and
  mRMR is refitted inside every training fold.
- **Paper's feature counts:** 80 (rest), 47 (music), 103 (rest+music), all 171 on DS-2.

### 5.5 KNN classifier ★★★
- **How:** a window is labelled by a vote of its k nearest training windows.
- **Paper's Table II settings:** k = 1 or 2; distances cityblock (sum of absolute
  differences), Euclidean or cosine; equal votes or votes weighted by 1/distance².
- **Why it fails on new people:** with k = 1–2, the nearest window is almost always another
  window of the same person, which the person-identification probe shows directly.

---

## 6. The evaluation protocol (your main contribution) ★★★

### 6.1 Segment-wise vs subject-wise splitting
- **Segment-wise (the paper's protocol):** all windows are shuffled into 10 folds, so windows
  of the same person, even overlapping ones, are in both training and test. The model can
  recognise the person instead of the disorder. It answers "can the model classify new
  windows of people it already knows?"
- **Subject-wise (your protocol):** all windows of a person stay on one side. It answers "does
  it work on a new person?", which is the clinical question.
- **DS-1:** leave-one-subject-out (LOSO): train on 13, test on 1, repeat 14 times.
- **DS-2:** stratified grouped 10-fold: 10 groups of children with balanced classes and a
  fixed seed.

### 6.2 Fitting everything inside the training fold
Scaling, mRMR, channel selection, hyperparameter tuning and early stopping only ever see
training people. Fitting any of them on all data leaks information about the test people.

### 6.3 Nested cross-validation
- **Outer loop:** tests.
- **Inner loop:** inside each outer training set, a grouped 4-fold split chooses the settings
  (for example k, the distance and the number of features).
- **Result for the paper's KNN:** 43.8% on DS-1 rest and
  65.3% on DS-2. Tuning changes from fold to fold because there are so few
  training people.

### 6.4 The automated leakage test
`python -m pytest tests` checks that no person is in both training and test in any fold,
including the inner tuning folds, and that the live pipeline gives the same windows and
features as the offline code.

### 6.5 Person-level decisions
Each person is classified by majority vote over their windows. "Child-level" in the report is a
metric name meaning "per person"; for DS-1 the people are adults.

### 6.6 Split variance
Over 20 random assignments of DS-2 children to folds, subject-wise accuracy is
59.0 ± 0.5% (range 57.7–60.4%). The
mid-semester number (57.54%) used an unshuffled split.

**Likely question:** "How do you know your own evaluation doesn't leak?" Answer: the split
design, fitting inside folds, the automated test, and the permutation test (section 12.1).

---

## 7. Statistics used in evaluation ★★★

| Term | Meaning |
|---|---|
| Accuracy | fraction correct |
| Sensitivity | fraction of patients detected: TP / (TP + FN) |
| Specificity | fraction of controls correctly cleared: TN / (TN + FP) |
| Balanced accuracy | (sensitivity + specificity) / 2; not fooled by unequal class sizes |
| F1 | harmonic mean of precision and sensitivity |
| ROC curve | sensitivity against false-positive rate as the threshold moves |
| AUC | chance that a random patient scores higher than a random control: 0.5 is chance, 1 perfect, below 0.5 reversed |
| Confusion matrix | table of true against predicted classes |

### Bootstrap confidence intervals over people
- Resample people with replacement 2000 times (keeping the class balance), recompute the
  metric each time, and take the middle 95% of values.
- **Why people, not windows:** windows of one person are not independent; resampling windows
  would give intervals that are far too narrow.
- **Limit:** if every person is correct, every resample is also 100%, so the interval
  "100–100%" is meaningless. The exact (Clopper–Pearson) interval for 14 out of 14 is
  76.8–100%.

### Permutation test
Shuffle the labels between people, rerun the whole pipeline, and see whether accuracy drops to
chance. If it stays high, the pipeline is leaking. Details in section 12.1.

### Selection bias
Choosing the best of many methods using the same test folds makes the winner's score slightly
optimistic. The paper's fixed Table II model and the nested-tuned KNN are not affected.

---

## 8. The other feature sets (Stages 2 and 2b)

### 8.1 Spectral features ★★★
Per channel, 16 numbers: log band power in 5 bands, relative band power in 5 bands, the log
theta/beta ratio, three Hjorth parameters, spectral entropy and sample entropy.
- **Hjorth activity** = variance (signal size); **mobility** ≈ mean frequency; **complexity** ≈
  how irregular the waveform is (1 for a pure sine).
- **Spectral entropy:** how spread out the power spectrum is.
- **Sample entropy:** how unpredictable the signal is (chance that patterns that match for 2
  samples still match for 3).
- Used by the demo model and the SHAP analysis.

### 8.2 Connectivity features ★★
Per channel pair, in theta, alpha and beta:
- **Coherence:** how consistently two channels oscillate together (0 to 1).
- **wPLI (weighted phase lag index):** phase coupling that ignores zero-lag coupling, which is
  mostly volume conduction (one source seen by both electrodes).

### 8.3 Filter-bank log-variance ★★
Filter into 6 bands (1–4, 4–8, 8–13, 13–20, 20–30, 30–45 Hz) and take the log of each channel's
variance in each band: a simple, classic EEG feature.

### 8.4 Wavelet-packet energies ★★
Wavelet packet decomposition (db4, level 4) gives 16 equal 4 Hz sub-bands; features are the
log energy and relative energy per sub-band and channel.

### 8.5 Aperiodic (1/f) features and entropies ★★
- EEG power falls with frequency roughly as 1/f^χ. A straight-line fit of log power against
  log frequency (2–25 Hz, skipping 7–14 Hz so the alpha peak does not bias it) gives the
  exponent χ and an offset.
- Plus spectral, sample and permutation entropy (permutation entropy counts up/down patterns).

### 8.6 Riemannian tangent space ★★★ (your best DS-2 method)
- **Step 1:** each window becomes a covariance matrix: each channel's power on the diagonal and
  how channel pairs vary together off the diagonal. Your code uses OAS shrinkage, which pulls
  the estimate slightly towards a scaled identity matrix to keep it stable.
- **Step 2:** covariance matrices live on a curved space, so straight-line averages and
  distances distort them. For example, averaging diag(1, 100) and diag(100, 1), both with
  determinant 100, gives a matrix with determinant 2550.
- **Step 3:** measure distance along the curved space with the Riemannian distance, which does
  not change if all channels are rescaled or mixed in the same way.
- **Step 4:** flatten the space around the average matrix (the tangent space), like a flat
  map of a small part of the globe, and turn each matrix into a vector: 190 numbers for 19
  channels.
- **Step 5:** classify the vectors with logistic regression or an SVM.
- **Result:** Riemannian tangent space (OAS covariance) + SVM, 72.6% balanced accuracy (CI
  67.4–77.2%), 92/121 children correct.

### 8.7 Transductive methods (marked †) ★★
These use the test person's *unlabelled* windows at test time, so they are reported but never
chosen as best.
- **Euclidean alignment:** whiten each person's data with their own average covariance, so
  every person's average becomes the identity matrix.
- **Per-person z-scoring:** standardise features within each person. With logistic regression
  this gave exactly 50% (marked ‡): every person's features average zero, so the best weights
  are all zero and every window gets the same prediction.

### 8.8 Person-level aggregation ★
One vector per person (mean and standard deviation of spectral features over their windows)
with logistic regression, so window metrics equal person metrics.

---

## 9. The other models

### 9.1 Classic models ★★★
| Model | How it decides | Note |
|---|---|---|
| Logistic regression | weighted sum of features passed through a sigmoid to give a probability | strength C controls regularisation; the demo model uses C = 0.01 |
| SVM (RBF kernel) | finds the boundary with the widest margin; the kernel allows curved boundaries | C trades margin against errors |
| Random forest | many decision trees on random subsets of data and features; majority vote | reduces overfitting of single trees |
| XGBoost | trees added one at a time, each correcting the errors of the previous ones | used for SHAP explanations |
| Ridge classifier | least squares with a penalty on large weights | MiniRocket's classifier |

All are tuned by nested cross-validation with small grids, with features standardised inside
the training fold.

### 9.2 EEGNet ★★
A compact convolutional network for EEG: a temporal convolution (learned frequency filters),
then a depthwise spatial convolution (learned channel combinations), then a separable
convolution. Very few parameters.

### 9.3 ShallowConvNet ★★
Temporal convolution → spatial convolution → squaring → averaging → log → classifier. This is
a learned version of filter-bank log-variance. It reached 99.7% on DS-1
music.

**Training details for both:** early stopping on held-out training people, and augmentation on
training windows only (random time shift, amplitude scaling 0.9–1.1, small Gaussian noise).

### 9.4 MiniRocket ★★★
About 10,000 fixed, random, tiny convolution kernels (length 9) at many dilations. For each,
the feature is the fraction of positive outputs after adding a bias. A ridge classifier is
trained on these features. It is fast and strong for time series, and was best on DS-1 rest
(99.8% balanced accuracy).

### 9.5 Ensemble ★
The average predicted probability of five members fixed in advance: tuned KNN, spectral
logistic regression, spectral XGBoost, EEGNet and Riemannian logistic regression.

**Result across all methods:** on DS-1 rest, 13 of
21 inductive methods have a 95% interval above chance; on DS-2, all
21.

---

## 10. Ablations and extra experiments

### 10.1 Ablation ladder ★★★
Same nine features and the same integration, with the decomposition swapped: raw signal →
fixed bands → VMD → MVMD → SMVMD.

| Balanced accuracy | Raw | SMVMD |
|---|---|---|
| DS-1 rest, paper KNN | 66.1% | 62.7% |
| DS-1 music, paper KNN | 67.5% | 43.0% |
| DS-2, logistic regression | 64.6% | 61.1% |

**Finding:** the raw signal does as well as SMVMD or better, so the decomposition is not what
makes classification work. The full table is in the report.

### 10.2 Number of mRMR features, window length, overlap ★
Accuracy against how many features mRMR keeps; window length 2.5, 5 and 10 s; overlap 0, 50
and 75%. These show how sensitive results are to the paper's choices.

### 10.3 Cross-condition (DS-1) ★
Train on rest and test on music, and the reverse, still leaving people out.

### 10.4 Cross-dataset and dataset identification ★★
- Use the 10 electrodes both datasets share (F7, F3, F4, F8, T7, T8, P7, P8, O1, O2) and scale-free features.
- Train on one dataset and test on the other: results are close to chance. This measures
  domain shift (different disorder, headset, task, age), not disorder transfer.
- A classifier tells DS-1 windows from DS-2 windows with 99.8%
  balanced accuracy: recording setup leaves a strong fingerprint.

### 10.5 Channel reduction ★★
Channels are ranked inside each training fold by how well each one alone predicts the label;
then the top 1, 2, … are used. On DS-1 rest: 1 channel 76.4% (CI
59.0–90.5), 4 channels 72.4%, all 68.0%. The
intervals overlap almost completely, so this does not show fewer channels are better, only
that a few channels can carry most of the signal.

---

## 11. Explaining the models

### 11.1 SHAP ★★
- **What:** Shapley values from game theory: each feature's average contribution to a
  prediction, over all combinations of the other features. They add up to the prediction
  minus the average prediction.
- **How used:** on the spectral XGBoost model, computed on test windows only.
- **Top features:** DS-1 rest logpow_beta, relpow_beta, hjorth_activity (channels F4, T8, AF3); DS-2
  logpow_gamma, hjorth_activity, relpow_gamma (channels P4, Pz, F4).
- **Caution:** importance shows what the model uses in these data, not a brain mechanism.
  Beta and gamma power are also raised by muscle activity.

### 11.2 Theta/beta ratio ★★★
- Scored without training, with the direction fixed in advance (higher ratio = disorder).
- DS-2: AUC 0.42 (CI 0.36–0.49): the ADHD group has a *lower* ratio, the
  opposite of the classic hypothesis. Stimulant medication is reported to lower it, so this
  fits the medication confound, but the data cannot prove that.
- DS-1 rest: AUC 0.38 (CI 0.18–0.60), chance.

### 11.3 EEGNet attribution ★
- Gradient × input per channel gives a scalp map; band occlusion removes one band with a
  band-stop filter and measures the drop in the probability of the true class.
- Mean probability of the true class without removal: 0.68 on DS-1 rest
  (beta mattered most), 0.55 on DS-2 (too close to chance to interpret).

### 11.4 t-SNE, silhouette and the person-identification probe ★★★
- **t-SNE:** squeezes high-dimensional features into 2-D while keeping neighbours together;
  distances between far-apart clusters mean nothing.
- **Silhouette score** (−1 to 1): how well points cluster by a given label. On DS-1 rest the
  features cluster more by person (0.218) than by class
  (0.075).
- **Probe:** a 1-nearest-neighbour classifier that predicts *which person* a window came from
  gets 98.91% (chance 7.14%). This is the core evidence
  that the features identify the individual.

---

## 12. Checks for leaks and confounds ★★★

### 12.1 Permutation test
- ShallowConvNet gets 99.7% on DS-1 music under leave-one-subject-out.
- With the participants' labels shuffled (still 7 against 7) and everything else identical,
  it gets 42.1–47.7%, which is chance.
- So the pipeline does not leak: it can only score high when the labels match a real
  difference between the groups.
- Limit: two permutations, so it is a sanity check rather than a significance test.

### 12.2 Why DS-1 reaches about 100%
- One number per recording, relative beta power, separates all 14 rest recordings: IDD
  0.139–0.286, controls 0.040–0.134 (AUC 1.000). On
  music it separates them partly (AUC 0.857).
- Possible causes: a real IDD effect, muscle tension, recording session or electrode contact,
  or age. Seven people per group cannot separate them.
- One control rest recording (CGS01_Rest) has 10 times the median
  amplitude, a large artefact left in the dataset's cleaned files.

### 12.3 DS-2 confounds
- Medication: ADHD children on methylphenidate, controls not.
- Behaviour: ADHD children took longer on the task; recording length alone predicts the label
  with AUC 0.656 (CI 0.554–0.759).

### 12.4 Audit of all results
Every saved metric was recomputed from the saved predictions and matched; no person is in two
test folds; no person has two labels.

---

## 13. The live demo ★★★
- **Pipeline:** source (replay of a recording, or a headset over Lab Streaming Layer, LSL) →
  ring buffer → causal 1–30 Hz band-pass and 50 Hz notch → 5 s windows every 2.5 s → spectral
  features → logistic regression → probability of IDD, smoothed over the last 4 windows.
- **Model:** spectral features + logistic regression, trained on all 14 DS-1 rest
  participants. Chosen in advance for speed and explainability, not because it scored
  highest.
- **Its accuracy on new people (subject-wise, same recipe):** 67.4% of windows (CI
  48.3–85.4%), 10/14 participants correct. This is what the dashboard
  shows.
- **What it learned:** frontal beta and theta push towards IDD; temporal and parietal alpha
  push towards control.
- **Equivalence test:** replaying a recording through the live path in irregular chunks gives
  the same windows and features as the offline code.
- **Limits:** replayed people were in the training data, so replay is not a test. A live
  headset differs in device, session and cleaning (no ICA live). Jaw clenching raises frontal
  beta and pushes the output towards "IDD", so the output is not a diagnosis.
- **Why not MiniRocket in the demo:** its DS-1 score is probably driven by group differences
  that a live recording will not share, and it cannot be explained feature by feature.

---

## 14. Key numbers to memorise ★★★

| What | Number |
|---|---|
| Paper's claim | 100% (DS-1), 99.17% (DS-2) |
| Reproduction, window-wise | 99.69% (DS-1 rest), 85.81% (DS-2) |
| Same model, subject-wise | 62.73% (10/14), 59.30% (78/121) |
| DS-1 music, subject-wise | 43.01% |
| Person-identification probe | 98.91% (chance 7.14%) |
| Best DS-2 method | 72.6% balanced, 92/121 children |
| Best DS-1 rest method | 99.8% balanced, 14/14 |
| Exact CI for 14/14 | 76.8–100% |
| Permutation test | 42.1–47.7% |
| Dataset identification | 99.8% |
| Theta/beta AUC on DS-2 | 0.42 |
| Demo model | 67.4% of windows, 10/14 |

---

## 15. Self-test
Answer out loud, then check against the sections above and `docs/panel_qa.md`.

1. Why does window-wise splitting overestimate accuracy? (6.1, 11.4)
2. What does α do in VMD, and why does K vary in SMVMD? (5.1)
3. Why is energy-weighted integration needed? (5.3)
4. What does mRMR maximise? (5.4)
5. Why bootstrap over people instead of windows? (7)
6. Why is "100–100%" not a real confidence interval? (7)
7. What does the permutation test prove, and what does it not? (12.1)
8. Why might DS-1 reach 100% without detecting IDD? (12.2)
9. Why is the DS-2 theta/beta result interesting? (11.2)
10. What does the ablation ladder show about SMVMD? (10.1)
11. Explain the Riemannian method in four steps. (8.6)
12. Why are methods marked † excluded from "best"? (8.7)
13. Why did per-person z-scored logistic regression give exactly 50%? (8.7)
14. Why did you pick a spectral logistic-regression model for the demo? (13)
15. What would you need to make a real diagnostic system? (an independent cohort recorded the
    same way for both groups, medication controlled, live artefact removal, clinical
    validation)
