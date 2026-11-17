# Stage 3: window ablation (subject-wise)

Generated from the result JSON files listed in the CSV `source` column. Percentages; 95% CIs are percentile bootstrap over children (stratified by class).


## DS1_Rest

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| spectral | Logistic regression (nested C) | subject_loso | 67.4 [48–85] | 67.4 [48–85] | 0.652 [0.40–0.89] | 71.4 [50–93] | 10/14 | 0.694 [0.37–0.96] | 0 | 2 |
| spectral | Logistic regression (nested C) | subject_loso | 66.3 [48–84] | 66.3 [48–84] | 0.632 [0.38–0.87] | 71.4 [50–93] | 10/14 | 0.694 [0.37–1.00] | 0 | 3 |
| spectral | Logistic regression (nested C) | subject_loso | 66.2 [47–83] | 66.2 [47–83] | 0.644 [0.37–0.93] | 71.4 [50–93] | 10/14 | 0.694 [0.39–1.00] | 1 | 1 |
| spectral | Logistic regression (nested C) | subject_loso | 61.8 [45–78] | 61.8 [45–78] | 0.654 [0.43–0.85] | 64.3 [43–86] | 9/14 | 0.673 [0.37–0.96] | 0 | 8 |
| spectral | Logistic regression (nested C) | subject_loso | 60.9 [43–79] | 60.9 [43–79] | 0.610 [0.36–0.84] | 71.4 [50–93] | 10/14 | 0.633 [0.33–0.92] | 0 | 2 |

## DS1_Music

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| spectral | Logistic regression (nested C) | subject_loso | 71.2 [56–86] | 71.2 [56–86] | 0.733 [0.49–0.93] | 85.7 [71–100] | 12/14 | 0.837 [0.55–1.00] | 0 | 3 |
| spectral | Logistic regression (nested C) | subject_loso | 69.8 [52–87] | 69.8 [52–87] | 0.769 [0.54–0.94] | 71.4 [50–93] | 10/14 | 0.796 [0.51–1.00] | 0 | 3 |
| spectral | Logistic regression (nested C) | subject_loso | 68.9 [50–87] | 68.9 [50–87] | 0.741 [0.50–0.94] | 64.3 [36–86] | 9/14 | 0.816 [0.55–1.00] | 0 | 2 |
| spectral | Logistic regression (nested C) | subject_loso | 67.5 [47–87] | 67.5 [47–87] | 0.714 [0.46–0.92] | 64.3 [43–86] | 9/14 | 0.755 [0.45–0.96] | 1 | 2 |
| spectral | Logistic regression (nested C) | subject_loso | 66.9 [48–86] | 66.9 [48–86] | 0.732 [0.48–0.94] | 57.1 [29–86] | 8/14 | 0.776 [0.49–1.00] | 0 | 2 |

## DS2

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| spectral | Logistic regression (nested C) | subject_sgkf | 63.9 [58–69] | 63.8 [58–69] | 0.680 [0.61–0.75] | 68.6 [60–77] | 83/121 | 0.727 [0.63–0.82] | 1 | 16 |
| spectral | Logistic regression (nested C) | subject_sgkf | 62.7 [57–68] | 62.7 [57–68] | 0.678 [0.61–0.74] | 67.8 [59–75] | 82/121 | 0.749 [0.65–0.83] | 2 | 16 |
| spectral | Logistic regression (nested C) | subject_sgkf | 62.3 [57–68] | 62.3 [57–68] | 0.672 [0.60–0.74] | 66.9 [59–75] | 81/121 | 0.732 [0.64–0.82] | 0 | 27 |
| spectral | Logistic regression (nested C) | subject_sgkf | 61.8 [56–67] | 61.7 [56–67] | 0.663 [0.59–0.73] | 66.9 [59–75] | 81/121 | 0.720 [0.62–0.81] | 0 | 67 |
| spectral | Logistic regression (nested C) | subject_sgkf | 61.6 [57–66] | 61.7 [57–66] | 0.664 [0.60–0.73] | 68.6 [60–76] | 83/121 | 0.749 [0.66–0.83] | 0 | 48 |

(†) uses unlabelled data of the test child at test time (transductive); not comparable to strictly inductive rows.

DS-2: ADHD children were taking methylphenidate and the controls were not; ADHD cannot be separated from its treatment in this dataset.