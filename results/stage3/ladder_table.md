# Stage 3: ladder ablation (subject-wise)

Generated from the result JSON files listed in the CSV `source` column. Percentages; 95% CIs are percentile bootstrap over children (stratified by class).


## DS1_Rest

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| bands + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 66.9 [47–84] | 66.9 [47–84] | 0.686 [0.48–0.88] | 64.3 [43–86] | 9/14 | 0.724 [0.41–0.96] | 0 | 381 |
| raw + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 66.1 [47–83] | 66.1 [47–83] | 0.671 [0.47–0.85] | 64.3 [43–86] | 9/14 | 0.673 [0.35–0.93] | 0 | 428 |
| smvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 62.7 [44–80] | 62.7 [44–80] | 0.618 [0.41–0.81] | 71.4 [50–93] | 10/14 | 0.602 [0.27–0.89] | 0 | 3 |
| vmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 61.5 [43–78] | 61.5 [43–78] | 0.651 [0.44–0.84] | 57.1 [36–79] | 8/14 | 0.633 [0.31–0.91] | 0 | 384 |
| raw + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 61.2 [41–81] | 61.2 [41–81] | 0.565 [0.29–0.84] | 64.3 [36–86] | 9/14 | 0.592 [0.27–0.90] | 0 | 4 |
| vmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 59.2 [39–78] | 59.2 [39–78] | 0.587 [0.31–0.84] | 71.4 [50–93] | 10/14 | 0.653 [0.33–0.94] | 1 | 4 |
| mvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 57.9 [40–75] | 57.9 [40–75] | 0.577 [0.37–0.77] | 57.1 [36–79] | 8/14 | 0.551 [0.21–0.88] | 0 | 378 |
| bands + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 56.5 [36–75] | 56.5 [36–75] | 0.534 [0.27–0.81] | 57.1 [29–79] | 8/14 | 0.592 [0.29–0.92] | 1 | 4 |
| mvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 51.4 [31–71] | 51.4 [31–71] | 0.510 [0.23–0.78] | 57.1 [36–79] | 8/14 | 0.551 [0.22–0.86] | 0 | 4 |
| smvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 48.4 [28–68] | 48.4 [28–68] | 0.497 [0.21–0.79] | 42.9 [21–64] | 6/14 | 0.510 [0.18–0.84] | 1 | 4 |

## DS1_Music

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| raw + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 67.5 [51–84] | 67.5 [51–84] | 0.710 [0.53–0.87] | 57.1 [29–86] | 8/14 | 0.837 [0.59–1.00] | 0 | 379 |
| vmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 65.8 [47–84] | 65.8 [47–84] | 0.654 [0.39–0.88] | 71.4 [50–93] | 10/14 | 0.653 [0.35–0.94] | 0 | 1 |
| mvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 64.4 [43–83] | 64.4 [43–83] | 0.605 [0.34–0.85] | 64.3 [36–86] | 9/14 | 0.612 [0.29–0.90] | 0 | 1 |
| smvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 62.4 [42–81] | 62.4 [42–81] | 0.612 [0.35–0.85] | 64.3 [36–86] | 9/14 | 0.592 [0.27–0.90] | 0 | 1 |
| raw + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 61.2 [42–79] | 61.2 [42–79] | 0.697 [0.42–0.91] | 57.1 [29–79] | 8/14 | 0.735 [0.41–0.96] | 0 | 4 |
| bands + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 59.9 [39–79] | 59.9 [39–79] | 0.680 [0.42–0.89] | 57.1 [29–79] | 8/14 | 0.714 [0.41–0.96] | 0 | 5 |
| vmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 57.8 [42–72] | 57.8 [42–72] | 0.592 [0.40–0.75] | 50.0 [21–72] | 7/14 | 0.694 [0.37–0.96] | 0 | 4370 |
| mvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 52.0 [35–68] | 52.0 [35–68] | 0.518 [0.33–0.70] | 57.1 [29–79] | 8/14 | 0.531 [0.20–0.86] | 0 | 204 |
| bands + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 51.9 [32–70] | 51.9 [32–70] | 0.534 [0.32–0.74] | 64.3 [36–86] | 9/14 | 0.469 [0.16–0.80] | 0 | 376 |
| smvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 43.0 [28–58] | 43.0 [28–58] | 0.410 [0.23–0.59] | 57.1 [29–79] | 8/14 | 0.327 [0.06–0.65] | 0 | 1 |

## DS1_RestMusic

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| raw + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 67.6 [50–84] | 67.6 [50–84] | 0.676 [0.50–0.84] | 71.4 [50–93] | 10/14 | 0.755 [0.47–0.96] | 1 | 454 |
| raw + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 66.5 [48–83] | 66.5 [48–83] | 0.753 [0.51–0.95] | 64.3 [36–86] | 9/14 | 0.776 [0.47–1.00] | 0 | 2 |
| vmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 65.7 [47–82] | 65.7 [47–82] | 0.657 [0.47–0.82] | 64.3 [43–86] | 9/14 | 0.776 [0.48–1.00] | 0 | 600 |
| bands + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 65.1 [46–82] | 65.1 [46–82] | 0.651 [0.46–0.82] | 64.3 [43–86] | 9/14 | 0.776 [0.47–0.98] | 0 | 592 |
| vmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 63.5 [44–81] | 63.5 [44–81] | 0.609 [0.35–0.88] | 71.4 [50–93] | 10/14 | 0.633 [0.31–1.00] | 0 | 2 |
| mvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 62.6 [43–79] | 62.6 [43–79] | 0.652 [0.40–0.87] | 71.4 [50–93] | 10/14 | 0.694 [0.37–0.96] | 0 | 2 |
| mvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 62.5 [43–80] | 62.5 [43–80] | 0.625 [0.43–0.80] | 64.3 [43–86] | 9/14 | 0.684 [0.37–0.94] | 0 | 596 |
| smvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_loso | 62.3 [44–80] | 62.3 [44–80] | 0.623 [0.44–0.80] | 64.3 [43–86] | 9/14 | 0.684 [0.36–0.94] | 0 | 1 |
| bands + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 60.5 [41–79] | 60.5 [41–79] | 0.641 [0.38–0.86] | 57.1 [29–79] | 8/14 | 0.714 [0.39–1.00] | 0 | 2 |
| smvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_loso | 58.9 [40–76] | 58.9 [40–76] | 0.581 [0.31–0.84] | 71.4 [50–93] | 10/14 | 0.612 [0.29–0.96] | 0 | 2 |

## DS2

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| raw + 9 feats + Eq.7 | Logistic regression (nested C) | subject_sgkf | 64.6 [59–70] | 64.6 [59–70] | 0.680 [0.61–0.75] | 66.9 [58–75] | 81/121 | 0.753 [0.66–0.83] | 1 | 19 |
| vmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_sgkf | 63.2 [58–68] | 63.2 [58–68] | 0.668 [0.60–0.73] | 69.4 [61–78] | 84/121 | 0.755 [0.66–0.83] | 0 | 19 |
| mvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_sgkf | 61.5 [57–66] | 61.4 [56–66] | 0.648 [0.58–0.71] | 65.3 [57–74] | 79/121 | 0.715 [0.62–0.80] | 2 | 19 |
| bands + 9 feats + Eq.7 | Logistic regression (nested C) | subject_sgkf | 61.1 [56–66] | 61.2 [56–66] | 0.647 [0.57–0.72] | 62.8 [54–71] | 76/121 | 0.712 [0.61–0.80] | 0 | 19 |
| smvmd + 9 feats + Eq.7 | Logistic regression (nested C) | subject_sgkf | 61.1 [56–66] | 61.1 [56–66] | 0.642 [0.57–0.71] | 66.1 [58–74] | 80/121 | 0.707 [0.61–0.79] | 1 | 19 |
| vmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_sgkf | 59.7 [57–62] | 59.4 [57–62] | 0.620 [0.58–0.65] | 72.7 [65–80] | 88/121 | 0.799 [0.72–0.87] | 5 | 4 |
| smvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_sgkf | 59.3 [56–62] | 58.9 [56–62] | 0.615 [0.57–0.65] | 64.5 [56–73] | 78/121 | 0.732 [0.64–0.81] | 4 | 4 |
| mvmd + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_sgkf | 58.4 [56–61] | 58.0 [55–61] | 0.607 [0.57–0.64] | 66.1 [58–74] | 80/121 | 0.750 [0.66–0.83] | 2 | 4 |
| raw + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_sgkf | 58.3 [55–61] | 57.9 [55–61] | 0.600 [0.56–0.64] | 61.2 [53–69] | 74/121 | 0.703 [0.61–0.79] | 3 | 4 |
| bands + 9 feats + Eq.7 | KNN (paper Table II) + mRMR | subject_sgkf | 57.7 [55–60] | 57.3 [54–60] | 0.598 [0.56–0.63] | 64.5 [56–73] | 78/121 | 0.722 [0.63–0.80] | 1 | 5 |

(†) uses unlabelled data of the test child at test time (transductive); not comparable to strictly inductive rows.

DS-2: ADHD children were taking methylphenidate and the controls were not; ADHD cannot be separated from its treatment in this dataset.