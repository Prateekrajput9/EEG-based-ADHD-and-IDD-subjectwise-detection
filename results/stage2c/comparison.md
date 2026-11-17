# Stage 2c: SMVMD common modes + Riemannian tangent space

Generated from the result JSON files listed in the CSV `source` column. Percentages; 95% CIs are percentile bootstrap over children (stratified by class).

Per-mode rows are an analysis; the "chosen by inner CV" rows pick the mode inside the training folds and are the leakage-free answer. Step G rows are included for reference.


## DS1_Rest

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Butterworth bands (all four), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 74 |
| Butterworth band (alpha), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 22 |
| Butterworth band (beta), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| SMVMD common modes (all four) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 72 |
| SMVMD common mode (alpha) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 36 |
| SMVMD common mode (beta) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 22 |
| broadband covariance, no decomposition (Step G input) | Tangent space + XGBoost | subject_loso | 99.8 [100–100] | 99.8 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 32 |
| broadband covariance, no decomposition (Step G input) | Tangent space + LightGBM | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 14 |
| SMVMD common modes (all four) | Tangent space + XGBoost | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 110 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + XGBoost | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 153 |
| Butterworth band (alpha), control | Tangent space + SVM | subject_loso | 98.9 [97–100] | 98.9 [97–100] | 0.998 [0.99–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 21 |
| Butterworth band (theta), control | Tangent space + LR | subject_loso | 98.9 [97–100] | 98.9 [97–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 23 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LightGBM | subject_loso | 98.6 [96–100] | 98.6 [96–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 92 |
| SMVMD common modes (all four) | Tangent space + LightGBM | subject_loso | 98.4 [96–100] | 98.4 [96–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 54 |
| SMVMD reconstruction (all modes) | Tangent space + LR | subject_loso | 98.0 [94–100] | 98.0 [94–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 20 |
| broadband covariance (Step G reference) | Tangent space + LR | subject_loso | 98.0 [94–100] | 98.0 [94–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 1135 |
| SMVMD common mode (alpha) | Tangent space + SVM | subject_loso | 97.8 [94–100] | 97.8 [94–100] | 0.999 [0.99–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 21 |
| SMVMD common mode (delta) | Tangent space + SVM | subject_loso | 97.4 [93–100] | 97.4 [93–100] | 0.996 [0.98–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + SVM | subject_loso | 97.4 [93–100] | 97.4 [93–100] | 0.996 [0.98–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 129 |
| SMVMD common mode (theta) | Tangent space + LR | subject_loso | 97.4 [94–100] | 97.4 [94–100] | 0.998 [0.99–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 20 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LR | subject_loso | 97.2 [92–100] | 97.2 [92–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 150 |
| SMVMD common mode (delta) | Tangent space + LR | subject_loso | 95.8 [89–100] | 95.8 [89–100] | 0.997 [0.98–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 44 |
| Butterworth band (delta), control | Tangent space + LR | subject_loso | 95.0 [87–100] | 95.0 [87–100] | 0.995 [0.97–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 21 |
| Butterworth band (theta), control | Tangent space + SVM | subject_loso | 94.9 [86–100] | 94.9 [86–100] | 0.997 [0.99–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 20 |
| SMVMD common modes (all four) | Tangent space + SVM | subject_loso | 94.3 [85–100] | 94.3 [85–100] | 0.999 [0.99–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 63 |
| Butterworth bands (all four), control | Tangent space + SVM | subject_loso | 93.8 [82–100] | 93.8 [82–100] | 0.998 [0.99–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 69 |
| SMVMD common mode (theta) | Tangent space + SVM | subject_loso | 93.2 [85–99] | 93.2 [85–99] | 0.977 [0.95–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 20 |
| SMVMD reconstruction (all modes) | Tangent space + SVM | subject_loso | 92.4 [78–100] | 92.4 [78–100] | 0.996 [0.98–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 18 |
| broadband covariance (Step G reference) | Tangent space + SVM | subject_loso | 92.4 [78–100] | 92.4 [78–100] | 0.995 [0.98–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 69 |
| Butterworth band (delta), control | Tangent space + SVM | subject_loso | 92.1 [81–100] | 92.1 [81–100] | 0.990 [0.95–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 22 |
| Butterworth band (beta), control | Tangent space + SVM | subject_loso | 85.7 [71–100] | 85.7 [71–100] | 0.995 [0.98–1.00] | 85.7 [71–100] | 12/14 | 1.000 [1.00–1.00] | 0 | 18 |
| SMVMD common mode (beta) | Tangent space + SVM | subject_loso | 85.7 [71–100] | 85.7 [71–100] | 0.995 [0.98–1.00] | 85.7 [71–100] | 12/14 | 1.000 [1.00–1.00] | 0 | 18 |

## DS1_Music

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Butterworth bands (all four), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 65 |
| Butterworth band (alpha), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 21 |
| Butterworth band (alpha), control | Tangent space + SVM | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 22 |
| Butterworth band (beta), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 17 |
| Butterworth band (delta), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 18 |
| Butterworth band (theta), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| SMVMD reconstruction (all modes) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 20 |
| SMVMD common modes (all four) | Tangent space + LightGBM | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 72 |
| SMVMD common modes (all four) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 66 |
| SMVMD common mode (alpha) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| SMVMD common mode (alpha) | Tangent space + SVM | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 22 |
| SMVMD common mode (beta) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 17 |
| SMVMD common mode (delta) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LightGBM | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 130 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 116 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + SVM | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 118 |
| broadband covariance (Step G reference) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 253 |
| Butterworth bands (all four), control | Tangent space + SVM | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 62 |
| Butterworth band (theta), control | Tangent space + SVM | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 20 |
| broadband covariance, no decomposition (Step G input) | Tangent space + XGBoost | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 35 |
| SMVMD reconstruction (all modes) | Tangent space + SVM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 25 |
| broadband covariance (Step G reference) | Tangent space + SVM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 374 |
| SMVMD common modes (all four) | Tangent space + SVM | subject_loso | 99.4 [98–100] | 99.4 [98–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 101 |
| Butterworth band (delta), control | Tangent space + SVM | subject_loso | 99.2 [98–100] | 99.2 [98–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| SMVMD common mode (delta) | Tangent space + SVM | subject_loso | 97.8 [96–100] | 97.8 [96–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 19 |
| broadband covariance, no decomposition (Step G input) | Tangent space + LightGBM | subject_loso | 97.7 [93–100] | 97.7 [93–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 18 |
| SMVMD common modes (all four) | Tangent space + XGBoost | subject_loso | 97.5 [93–100] | 97.5 [93–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 86 |
| SMVMD common mode (theta) | Tangent space + SVM | subject_loso | 96.6 [93–99] | 96.6 [93–99] | 0.977 [0.94–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 18 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + XGBoost | subject_loso | 95.2 [88–100] | 95.2 [88–100] | 0.995 [0.97–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 195 |
| SMVMD common mode (theta) | Tangent space + LR | subject_loso | 95.2 [91–99] | 95.2 [91–99] | 0.996 [0.99–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 17 |
| Butterworth band (beta), control | Tangent space + SVM | subject_loso | 93.2 [81–100] | 93.2 [81–100] | 1.000 [1.00–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 18 |
| SMVMD common mode (beta) | Tangent space + SVM | subject_loso | 92.9 [79–100] | 92.9 [79–100] | 1.000 [1.00–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 19 |

## DS1_RestMusic

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Butterworth bands (all four), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 141 |
| Butterworth band (alpha), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 42 |
| Butterworth band (beta), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 37 |
| Butterworth band (theta), control | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 37 |
| SMVMD reconstruction (all modes) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 37 |
| SMVMD common modes (all four) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 129 |
| SMVMD common modes (all four) | Tangent space + XGBoost | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 167 |
| SMVMD common mode (alpha) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 40 |
| SMVMD common mode (beta) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 35 |
| broadband covariance (Step G reference) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 154 |
| Butterworth bands (all four), control | Tangent space + SVM | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 137 |
| Butterworth band (alpha), control | Tangent space + SVM | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 44 |
| Butterworth band (theta), control | Tangent space + SVM | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 39 |
| SMVMD common mode (alpha) | Tangent space + SVM | subject_loso | 99.6 [99–100] | 99.6 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 41 |
| broadband covariance, no decomposition (Step G input) | Tangent space + LightGBM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 36 |
| SMVMD reconstruction (all modes) | Tangent space + SVM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 38 |
| SMVMD common mode (delta) | Tangent space + LR | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 36 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LR | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 230 |
| broadband covariance (Step G reference) | Tangent space + SVM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 172 |
| Butterworth band (delta), control | Tangent space + LR | subject_loso | 99.3 [98–100] | 99.3 [98–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 37 |
| SMVMD common modes (all four) | Tangent space + SVM | subject_loso | 99.3 [98–100] | 99.3 [98–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 160 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + XGBoost | subject_loso | 99.3 [98–100] | 99.3 [98–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 349 |
| broadband covariance, no decomposition (Step G input) | Tangent space + XGBoost | subject_loso | 98.8 [97–100] | 98.8 [97–100] | 0.999 [0.99–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 57 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + SVM | subject_loso | 98.6 [97–100] | 98.6 [97–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 236 |
| SMVMD common mode (delta) | Tangent space + SVM | subject_loso | 98.4 [96–100] | 98.4 [96–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 38 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LightGBM | subject_loso | 98.2 [96–100] | 98.2 [96–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 255 |
| SMVMD common mode (theta) | Tangent space + SVM | subject_loso | 98.0 [95–100] | 98.0 [95–100] | 0.993 [0.98–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 36 |
| Butterworth band (delta), control | Tangent space + SVM | subject_loso | 98.0 [95–100] | 98.0 [95–100] | 0.999 [0.99–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 39 |
| SMVMD common mode (theta) | Tangent space + LR | subject_loso | 97.3 [96–99] | 97.3 [96–99] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 34 |
| SMVMD common modes (all four) | Tangent space + LightGBM | subject_loso | 96.6 [90–100] | 96.6 [90–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 147 |
| Butterworth band (beta), control | Tangent space + SVM | subject_loso | 95.0 [85–100] | 95.0 [85–100] | 0.998 [0.99–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 37 |
| SMVMD common mode (beta) | Tangent space + SVM | subject_loso | 94.8 [84–100] | 94.8 [84–100] | 0.998 [0.99–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 34 |

## DS2

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| broadband covariance, no decomposition (Step G input) | Tangent space + LightGBM | subject_sgkf | 73.8 [69–78] | 73.1 [68–78] | 0.815 [0.76–0.86] | 76.0 [69–83] | 92/121 | 0.894 [0.83–0.94] | 1 | 162 |
| Butterworth bands (all four), control | Tangent space + SVM | subject_sgkf | 74.1 [69–79] | 72.8 [67–78] | 0.788 [0.72–0.85] | 74.4 [67–82] | 90/121 | 0.833 [0.76–0.90] | 1 | 1790 |
| SMVMD reconstruction (all modes) | Tangent space + SVM | subject_sgkf | 73.5 [69–78] | 72.6 [68–77] | 0.804 [0.75–0.85] | 76.0 [69–83] | 92/121 | 0.888 [0.83–0.94] | 2 | 449 |
| broadband covariance (Step G reference) | Tangent space + SVM | subject_sgkf | 73.5 [69–78] | 72.6 [67–77] | 0.803 [0.75–0.85] | 76.0 [69–83] | 92/121 | 0.884 [0.82–0.94] | 1 | 927 |
| broadband covariance, no decomposition (Step G input) | Tangent space + XGBoost | subject_sgkf | 73.0 [68–77] | 72.2 [68–77] | 0.803 [0.75–0.85] | 75.2 [69–82] | 91/121 | 0.873 [0.81–0.93] | 0 | 252 |
| SMVMD common modes (all four) | Tangent space + XGBoost | subject_sgkf | 72.2 [67–77] | 71.4 [66–76] | 0.785 [0.72–0.84] | 75.2 [68–83] | 91/121 | 0.837 [0.76–0.90] | 2 | 934 |
| SMVMD common modes (all four) | Tangent space + LightGBM | subject_sgkf | 72.2 [67–77] | 71.3 [66–76] | 0.786 [0.72–0.84] | 76.9 [69–83] | 93/121 | 0.833 [0.75–0.90] | 0 | 620 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LightGBM | subject_sgkf | 72.2 [67–77] | 71.3 [66–76] | 0.786 [0.72–0.84] | 76.9 [69–83] | 93/121 | 0.833 [0.75–0.90] | 0 | 1126 |
| broadband covariance (Step G reference) | Tangent space + LR | subject_sgkf | 71.5 [67–76] | 71.1 [66–76] | 0.786 [0.73–0.84] | 77.7 [70–84] | 94/121 | 0.867 [0.80–0.92] | 1 | 826 |
| SMVMD reconstruction (all modes) | Tangent space + LR | subject_sgkf | 71.3 [67–76] | 71.0 [66–75] | 0.785 [0.73–0.84] | 76.9 [69–83] | 93/121 | 0.869 [0.80–0.92] | 1 | 161 |
| Butterworth band (beta), control | Tangent space + SVM | subject_sgkf | 71.6 [67–76] | 71.0 [66–76] | 0.771 [0.71–0.83] | 77.7 [70–84] | 94/121 | 0.837 [0.76–0.90] | 1 | 395 |
| SMVMD common mode (delta) | Tangent space + SVM | subject_sgkf | 72.0 [67–76] | 70.9 [66–76] | 0.780 [0.72–0.83] | 74.4 [67–81] | 90/121 | 0.863 [0.79–0.92] | 1 | 499 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + XGBoost | subject_sgkf | 71.8 [67–77] | 70.9 [66–76] | 0.781 [0.72–0.84] | 75.2 [68–83] | 91/121 | 0.833 [0.75–0.90] | 2 | 1726 |
| SMVMD common modes (all four) | Tangent space + SVM | subject_sgkf | 71.6 [66–76] | 70.4 [65–75] | 0.772 [0.71–0.83] | 74.4 [67–81] | 90/121 | 0.818 [0.74–0.88] | 0 | 1887 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + SVM | subject_sgkf | 71.6 [66–76] | 70.4 [65–75] | 0.772 [0.71–0.83] | 74.4 [67–81] | 90/121 | 0.818 [0.74–0.88] | 0 | 3256 |
| Butterworth band (alpha), control | Tangent space + SVM | subject_sgkf | 71.0 [66–76] | 70.1 [65–75] | 0.759 [0.69–0.82] | 74.4 [67–82] | 90/121 | 0.800 [0.71–0.88] | 1 | 441 |
| Butterworth bands (all four), control | Tangent space + LR | subject_sgkf | 70.3 [65–75] | 70.0 [65–75] | 0.774 [0.71–0.83] | 74.4 [67–82] | 90/121 | 0.826 [0.75–0.89] | 3 | 628 |
| SMVMD common mode (delta) | Tangent space + LR | subject_sgkf | 70.0 [65–74] | 69.7 [65–74] | 0.766 [0.71–0.82] | 78.5 [71–85] | 95/121 | 0.854 [0.78–0.91] | 3 | 182 |
| Butterworth band (beta), control | Tangent space + LR | subject_sgkf | 69.6 [65–74] | 69.4 [64–74] | 0.759 [0.70–0.82] | 75.2 [68–83] | 91/121 | 0.835 [0.76–0.90] | 1 | 153 |
| SMVMD common mode (theta) | Tangent space + SVM | subject_sgkf | 70.1 [65–75] | 68.9 [63–74] | 0.735 [0.67–0.80] | 72.7 [65–80] | 88/121 | 0.767 [0.68–0.85] | 2 | 521 |
| Butterworth band (theta), control | Tangent space + SVM | subject_sgkf | 69.1 [64–74] | 68.2 [62–74] | 0.738 [0.67–0.80] | 72.7 [65–80] | 88/121 | 0.767 [0.67–0.85] | 1 | 434 |
| SMVMD common modes (all four) | Tangent space + LR | subject_sgkf | 68.3 [63–73] | 68.0 [63–73] | 0.750 [0.69–0.81] | 71.9 [64–79] | 87/121 | 0.813 [0.73–0.88] | 1 | 668 |
| Butterworth band (alpha), control | Tangent space + LR | subject_sgkf | 68.1 [63–73] | 67.9 [63–73] | 0.742 [0.68–0.80] | 71.1 [63–79] | 86/121 | 0.794 [0.71–0.87] | 1 | 161 |
| Butterworth band (theta), control | Tangent space + LR | subject_sgkf | 68.2 [63–73] | 67.9 [63–73] | 0.747 [0.68–0.81] | 70.2 [62–78] | 85/121 | 0.784 [0.70–0.86] | 1 | 176 |
| SMVMD common mode (theta) | Tangent space + LR | subject_sgkf | 67.8 [63–73] | 67.4 [62–73] | 0.744 [0.68–0.80] | 70.2 [62–78] | 85/121 | 0.783 [0.70–0.86] | 2 | 160 |
| SMVMD common mode chosen by inner CV (training subjects only) | Tangent space + LR | subject_sgkf | 67.8 [63–73] | 67.4 [62–72] | 0.741 [0.68–0.80] | 73.6 [65–81] | 89/121 | 0.807 [0.72–0.88] | 0 | 1100 |
| Butterworth band (delta), control | Tangent space + LR | subject_sgkf | 66.8 [62–71] | 66.6 [62–71] | 0.732 [0.67–0.79] | 74.4 [67–82] | 90/121 | 0.796 [0.71–0.87] | 3 | 193 |
| Butterworth band (delta), control | Tangent space + SVM | subject_sgkf | 67.0 [62–71] | 66.2 [61–71] | 0.729 [0.67–0.79] | 66.9 [59–74] | 81/121 | 0.796 [0.71–0.87] | 0 | 455 |
| SMVMD common mode (beta) | Tangent space + SVM | subject_sgkf | 62.2 [58–67] | 63.5 [59–68] | 0.643 [0.59–0.70] | 66.1 [59–74] | 80/121 | 0.790 [0.71–0.86] | 3 | 571 |
| SMVMD common mode (alpha) | Tangent space + LR | subject_sgkf | 62.6 [58–67] | 62.5 [58–67] | 0.691 [0.63–0.75] | 72.7 [64–80] | 88/121 | 0.772 [0.68–0.85] | 2 | 140 |
| SMVMD common mode (alpha) | Tangent space + SVM | subject_sgkf | 63.4 [59–68] | 62.5 [58–67] | 0.649 [0.59–0.71] | 68.6 [60–76] | 83/121 | 0.726 [0.63–0.81] | 1 | 552 |
| SMVMD common mode (beta) | Tangent space + LR | subject_sgkf | 58.9 [55–63] | 60.0 [56–64] | 0.659 [0.60–0.71] | 62.8 [55–71] | 76/121 | 0.782 [0.69–0.86] | 1 | 148 |

(†) uses unlabelled data of the test child at test time (transductive); not comparable to strictly inductive rows.

DS-2: ADHD children were taking methylphenidate and the controls were not; ADHD cannot be separated from its treatment in this dataset.