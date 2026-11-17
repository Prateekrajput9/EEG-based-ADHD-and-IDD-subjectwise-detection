# Stage 2b: accuracy-improvement sweep (all methods)

Generated from the result JSON files listed in the CSV `source` column. Percentages; 95% CIs are percentile bootstrap over children (stratified by class).

Includes Stage 1 (SMVMD) and Stage 2 rows for reference. (†) = transductive.


## DS1_Rest

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniRocket 10k kernels (sktime) | MiniRocket + ridge | subject_loso | 99.8 [100–100] | 99.8 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 108 |
| raw windows | ShallowConvNet | subject_loso | 99.1 [98–100] | 99.1 [98–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 462 |
| Riemannian tangent space (OAS covariance) | Tangent space + LR | subject_loso | 98.0 [94–100] | 98.0 [94–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 1135 |
| Riemannian tangent space (OAS covariance) | Tangent space + SVM | subject_loso | 92.4 [78–100] | 92.4 [78–100] | 0.995 [0.98–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 69 |
| connectivity | Logistic regression | subject_loso | 88.7 [77–97] | 88.7 [77–97] | 0.966 [0.91–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 10 |
| connectivity | Random forest | subject_loso | 85.2 [68–97] | 85.2 [68–97] | 0.928 [0.78–1.00] | 92.9 [79–100] | 13/14 | 0.959 [0.82–1.00] | 0 | 229 |
| connectivity | XGBoost | subject_loso | 84.9 [68–97] | 84.9 [68–97] | 0.913 [0.75–1.00] | 92.9 [79–100] | 13/14 | 0.898 [0.63–1.00] | 0 | 276 |
| connectivity | SVM (RBF) | subject_loso | 84.8 [71–95] | 84.8 [71–95] | 0.931 [0.81–0.99] | 85.7 [64–100] | 12/14 | 0.959 [0.82–1.00] | 0 | 17 |
| Euclidean-aligned raw windows | EEGNet-8,2 (†) | subject_loso | 78.9 [70–87] | 78.9 [70–87] | 0.847 [0.75–0.93] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 464 |
| spectral | XGBoost | subject_loso | 78.7 [66–90] | 78.7 [66–90] | 0.846 [0.69–0.96] | 92.9 [79–100] | 13/14 | 0.918 [0.69–1.00] | 0 | 193 |
| members fixed in advance | Ensemble (mean probability) | subject_loso | 76.2 [60–91] | 76.2 [60–91] | 0.893 [0.75–1.00] | 71.4 [50–93] | 10/14 | 0.918 [0.76–1.00] | 0 | 0 |
| spectral | Random forest | subject_loso | 71.7 [57–86] | 71.7 [57–86] | 0.717 [0.47–0.92] | 78.6 [57–100] | 11/14 | 0.776 [0.49–1.00] | 0 | 207 |
| spectral mean+SD per child | Logistic regression (child level) | subject_loso | 71.4 [50–93] | 71.4 [50–93] | 0.694 [0.37–1.00] | 71.4 [50–93] | 10/14 | 0.694 [0.37–1.00] | 0 | 14 |
| wpt | Logistic regression | subject_loso | 69.6 [52–86] | 69.6 [52–86] | 0.789 [0.56–0.95] | 71.4 [50–93] | 10/14 | 0.816 [0.53–1.00] | 0 | 5 |
| Euclidean-aligned covariance, tangent space | Tangent space + LR (†) | subject_loso | 69.1 [47–88] | 69.1 [47–88] | 0.676 [0.37–0.94] | 78.6 [57–100] | 11/14 | 0.694 [0.37–1.00] | 0 | 62 |
| raw windows | EEGNet-8,2 | subject_loso | 68.9 [53–85] | 68.9 [53–85] | 0.811 [0.57–0.98] | 71.4 [57–93] | 10/14 | 0.796 [0.51–1.00] | 0 | 1236 |
| fb_logvar | Logistic regression | subject_loso | 68.3 [50–85] | 68.3 [50–85] | 0.759 [0.52–0.94] | 71.4 [50–93] | 10/14 | 0.816 [0.53–1.00] | 1 | 4 |
| spectral | Logistic regression | subject_loso | 67.4 [48–85] | 67.4 [48–85] | 0.652 [0.40–0.89] | 71.4 [50–93] | 10/14 | 0.694 [0.37–0.96] | 0 | 72 |
| aperiodic | XGBoost | subject_loso | 64.4 [47–81] | 64.4 [47–81] | 0.649 [0.41–0.88] | 78.6 [57–100] | 11/14 | 0.694 [0.39–1.00] | 1 | 243 |
| aperiodic | Logistic regression | subject_loso | 64.0 [46–81] | 64.0 [46–81] | 0.648 [0.42–0.86] | 57.1 [36–86] | 8/14 | 0.755 [0.43–1.00] | 0 | 4 |
| SMVMD | SMVMD+mRMR+KNN (paper Table II) | subject_loso | 62.7 [44–80] | 62.7 [44–80] | 0.618 [0.41–0.81] | 71.4 [50–93] | 10/14 | 0.602 [0.27–0.89] | 0 | 255 |
| spectral | SVM (RBF) | subject_loso | 57.5 [38–78] | 57.5 [38–78] | 0.529 [0.25–0.81] | 57.1 [36–79] | 8/14 | 0.531 [0.20–0.86] | 0 | 11 |
| spectral, z-scored within each child | Logistic regression (†) | subject_loso | 50.0 [50–50] | 50.0 [50–50] | 0.500 [0.50–0.50] | 50.0 [50–50] | 7/14 | 0.500 [0.50–0.50] | 0 | 164 |
| SMVMD | SMVMD+mRMR+KNN (nested-tuned) | subject_loso | 43.8 [30–59] | 43.8 [30–59] | 0.433 [0.23–0.63] | 50.0 [29–71] | 7/14 | 0.367 [0.04–0.69] | 0 | 1741 |

## DS1_Music

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniRocket 10k kernels (sktime) | MiniRocket + ridge | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 95 |
| Riemannian tangent space (OAS covariance) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 253 |
| raw windows | ShallowConvNet | subject_loso | 99.7 [99–100] | 99.7 [99–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 304 |
| Riemannian tangent space (OAS covariance) | Tangent space + SVM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 374 |
| raw windows | EEGNet-8,2 | subject_loso | 93.2 [87–99] | 93.2 [87–99] | 0.980 [0.94–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 293 |
| connectivity | Logistic regression | subject_loso | 88.0 [73–98] | 88.0 [73–98] | 0.945 [0.80–1.00] | 92.9 [79–100] | 13/14 | 0.939 [0.76–1.00] | 0 | 11 |
| members fixed in advance | Ensemble (mean probability) | subject_loso | 87.0 [74–97] | 87.0 [74–97] | 0.961 [0.88–1.00] | 92.9 [79–100] | 13/14 | 0.980 [0.88–1.00] | 0 | 0 |
| connectivity | SVM (RBF) | subject_loso | 86.0 [70–97] | 86.0 [70–97] | 0.927 [0.77–1.00] | 92.9 [79–100] | 13/14 | 0.939 [0.76–1.00] | 0 | 18 |
| fb_logvar | Logistic regression | subject_loso | 81.2 [65–96] | 81.2 [65–96] | 0.850 [0.60–1.00] | 85.7 [71–100] | 12/14 | 0.857 [0.57–1.00] | 0 | 8 |
| connectivity | Random forest | subject_loso | 80.1 [61–95] | 80.1 [61–95] | 0.893 [0.71–0.99] | 78.6 [57–100] | 11/14 | 0.918 [0.69–1.00] | 0 | 241 |
| connectivity | XGBoost | subject_loso | 75.9 [56–93] | 75.9 [56–93] | 0.811 [0.57–0.97] | 78.6 [57–100] | 11/14 | 0.878 [0.63–1.00] | 0 | 255 |
| Euclidean-aligned raw windows | EEGNet-8,2 (†) | subject_loso | 75.9 [57–92] | 75.9 [57–92] | 0.819 [0.57–0.98] | 78.6 [57–100] | 11/14 | 0.837 [0.57–1.00] | 0 | 436 |
| wpt | Logistic regression | subject_loso | 73.8 [54–92] | 73.8 [54–92] | 0.823 [0.59–1.00] | 71.4 [43–93] | 10/14 | 0.878 [0.65–1.00] | 0 | 23 |
| spectral mean+SD per child | Logistic regression (child level) | subject_loso | 71.4 [50–93] | 71.4 [50–93] | 0.796 [0.51–1.00] | 71.4 [50–93] | 10/14 | 0.796 [0.51–1.00] | 0 | 10 |
| spectral | XGBoost | subject_loso | 70.5 [55–86] | 70.5 [55–86] | 0.791 [0.59–0.95] | 71.4 [50–93] | 10/14 | 0.837 [0.59–1.00] | 0 | 179 |
| aperiodic | Logistic regression | subject_loso | 68.3 [52–83] | 68.3 [52–83] | 0.773 [0.57–0.92] | 85.7 [64–100] | 12/14 | 0.857 [0.59–1.00] | 1 | 27 |
| spectral | Random forest | subject_loso | 68.0 [50–85] | 68.0 [50–85] | 0.711 [0.47–0.91] | 71.4 [50–93] | 10/14 | 0.776 [0.49–1.00] | 0 | 205 |
| spectral | Logistic regression | subject_loso | 66.9 [48–86] | 66.9 [48–86] | 0.732 [0.48–0.94] | 57.1 [29–86] | 8/14 | 0.776 [0.49–1.00] | 0 | 8 |
| SMVMD | SMVMD+mRMR+KNN (nested-tuned) | subject_loso | 63.8 [47–79] | 63.8 [47–79] | 0.640 [0.40–0.87] | 78.6 [57–100] | 11/14 | 0.714 [0.41–1.00] | 0 | 1349 |
| spectral | SVM (RBF) | subject_loso | 63.7 [47–80] | 63.7 [47–80] | 0.699 [0.46–0.90] | 64.3 [43–86] | 9/14 | 0.735 [0.45–0.96] | 0 | 10 |
| aperiodic | XGBoost | subject_loso | 63.7 [47–80] | 63.7 [47–80] | 0.672 [0.45–0.87] | 71.4 [50–93] | 10/14 | 0.694 [0.37–0.96] | 0 | 341 |
| Euclidean-aligned covariance, tangent space | Tangent space + LR (†) | subject_loso | 52.5 [33–71] | 52.5 [33–71] | 0.553 [0.27–0.81] | 57.1 [29–79] | 8/14 | 0.571 [0.24–0.88] | 0 | 231 |
| spectral, z-scored within each child | Logistic regression (†) | subject_loso | 50.0 [50–50] | 50.0 [50–50] | 0.500 [0.50–0.50] | 50.0 [50–50] | 7/14 | 0.500 [0.50–0.50] | 0 | 11 |
| SMVMD | SMVMD+mRMR+KNN (paper Table II) | subject_loso | 43.0 [28–58] | 43.0 [28–58] | 0.410 [0.23–0.59] | 57.1 [29–79] | 8/14 | 0.327 [0.06–0.65] | 0 | 433 |

## DS1_RestMusic

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| raw windows | ShallowConvNet | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 683 |
| MiniRocket 10k kernels (sktime) | MiniRocket + ridge | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 205 |
| Riemannian tangent space (OAS covariance) | Tangent space + LR | subject_loso | 100.0 [100–100] | 100.0 [100–100] | 1.000 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 154 |
| Riemannian tangent space (OAS covariance) | Tangent space + SVM | subject_loso | 99.5 [99–100] | 99.5 [99–100] | 0.999 [1.00–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 172 |
| connectivity | Logistic regression | subject_loso | 91.4 [83–98] | 91.4 [83–98] | 0.981 [0.94–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 9 |
| raw windows | EEGNet-8,2 | subject_loso | 90.5 [78–99] | 90.5 [78–99] | 0.995 [0.98–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 590 |
| connectivity | SVM (RBF) | subject_loso | 88.6 [79–96] | 88.6 [79–96] | 0.957 [0.88–1.00] | 100.0 [100–100] | 14/14 | 1.000 [1.00–1.00] | 0 | 33 |
| members fixed in advance | Ensemble (mean probability) | subject_loso | 87.8 [77–97] | 87.8 [77–97] | 0.975 [0.92–1.00] | 92.9 [79–100] | 13/14 | 1.000 [1.00–1.00] | 0 | 0 |
| connectivity | Random forest | subject_loso | 83.8 [67–96] | 83.8 [67–96] | 0.922 [0.78–0.99] | 92.9 [79–100] | 13/14 | 0.939 [0.76–1.00] | 0 | 298 |
| connectivity | XGBoost | subject_loso | 82.4 [65–95] | 82.4 [65–95] | 0.858 [0.64–0.99] | 85.7 [64–100] | 12/14 | 0.898 [0.63–1.00] | 0 | 226 |
| wpt | Logistic regression | subject_loso | 79.4 [65–92] | 79.4 [65–92] | 0.894 [0.75–0.98] | 78.6 [57–100] | 11/14 | 0.939 [0.80–1.00] | 0 | 8 |
| Euclidean-aligned raw windows | EEGNet-8,2 (†) | subject_loso | 78.9 [67–88] | 78.9 [67–88] | 0.838 [0.66–0.95] | 92.9 [79–100] | 13/14 | 0.939 [0.76–1.00] | 0 | 589 |
| spectral mean+SD per child | Logistic regression (child level) | subject_loso | 78.6 [57–100] | 78.6 [57–100] | 0.735 [0.41–1.00] | 78.6 [57–100] | 11/14 | 0.735 [0.41–1.00] | 0 | 2 |
| fb_logvar | Logistic regression | subject_loso | 78.4 [65–90] | 78.4 [65–90] | 0.880 [0.73–0.98] | 85.7 [64–100] | 12/14 | 0.918 [0.73–1.00] | 0 | 5 |
| spectral | XGBoost | subject_loso | 74.1 [58–88] | 74.1 [58–88] | 0.836 [0.66–0.96] | 78.6 [57–100] | 11/14 | 0.857 [0.61–1.00] | 0 | 128 |
| spectral | Logistic regression | subject_loso | 72.4 [58–86] | 72.4 [58–86] | 0.778 [0.60–0.93] | 78.6 [57–100] | 11/14 | 0.878 [0.67–1.00] | 0 | 10 |
| aperiodic | Logistic regression | subject_loso | 71.7 [56–86] | 71.7 [56–86] | 0.744 [0.52–0.93] | 78.6 [57–100] | 11/14 | 0.837 [0.55–1.00] | 0 | 4 |
| spectral | Random forest | subject_loso | 67.8 [50–83] | 67.8 [50–83] | 0.707 [0.46–0.92] | 71.4 [43–93] | 10/14 | 0.755 [0.45–1.00] | 1 | 249 |
| aperiodic | XGBoost | subject_loso | 64.2 [48–80] | 64.2 [48–80] | 0.658 [0.44–0.87] | 78.6 [57–100] | 11/14 | 0.673 [0.35–1.00] | 0 | 78 |
| SMVMD | SMVMD+mRMR+KNN (paper Table II) | subject_loso | 62.3 [44–80] | 62.3 [44–80] | 0.623 [0.44–0.80] | 64.3 [43–86] | 9/14 | 0.684 [0.36–0.94] | 0 | 857 |
| Euclidean-aligned covariance, tangent space | Tangent space + LR (†) | subject_loso | 56.4 [43–68] | 56.4 [43–68] | 0.586 [0.41–0.72] | 78.6 [57–100] | 11/14 | 0.735 [0.41–1.00] | 2 | 180 |
| SMVMD | SMVMD+mRMR+KNN (nested-tuned) | subject_loso | 55.8 [38–72] | 55.8 [38–72] | 0.564 [0.32–0.78] | 57.1 [29–79] | 8/14 | 0.592 [0.24–0.88] | 0 | 48917 |
| spectral | SVM (RBF) | subject_loso | 54.8 [37–73] | 54.8 [37–73] | 0.556 [0.29–0.82] | 64.3 [43–86] | 9/14 | 0.510 [0.16–0.84] | 0 | 13 |
| spectral, z-scored within each child | Logistic regression (†) | subject_loso | 50.0 [50–50] | 50.0 [50–50] | 0.500 [0.50–0.50] | 50.0 [50–50] | 7/14 | 0.500 [0.50–0.50] | 0 | 12 |

## DS2

| Features | Model | Protocol | Win acc [CI] | Win bal acc [CI] | Win AUC [CI] | Child acc [CI] | Children | Child AUC [CI] | Ties | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Riemannian tangent space (OAS covariance) | Tangent space + SVM | subject_sgkf | 73.5 [69–78] | 72.6 [67–77] | 0.803 [0.75–0.85] | 76.0 [69–83] | 92/121 | 0.884 [0.82–0.94] | 1 | 927 |
| Riemannian tangent space (OAS covariance) | Tangent space + LR | subject_sgkf | 71.5 [67–76] | 71.1 [66–76] | 0.786 [0.73–0.84] | 77.7 [70–84] | 94/121 | 0.867 [0.80–0.92] | 1 | 826 |
| connectivity | XGBoost | subject_sgkf | 70.4 [66–75] | 69.7 [65–74] | 0.757 [0.69–0.81] | 78.5 [71–85] | 95/121 | 0.829 [0.74–0.90] | 3 | 1293 |
| members fixed in advance | Ensemble (mean probability) | subject_sgkf | 70.2 [65–75] | 69.5 [64–75] | 0.759 [0.69–0.82] | 74.4 [67–82] | 90/121 | 0.815 [0.73–0.89] | 0 | 0 |
| spectral mean+SD per child | Logistic regression (child level) | subject_sgkf | 69.4 [61–78] | 69.4 [61–78] | 0.725 [0.63–0.81] | 69.4 [61–78] | 84/121 | 0.725 [0.63–0.81] | 0 | 2 |
| connectivity | SVM (RBF) | subject_sgkf | 68.7 [64–73] | 67.9 [63–72] | 0.741 [0.68–0.80] | 77.7 [70–84] | 94/121 | 0.822 [0.73–0.89] | 2 | 2738 |
| connectivity | Logistic regression | subject_sgkf | 67.9 [64–72] | 67.6 [63–72] | 0.730 [0.67–0.78] | 80.2 [73–87] | 97/121 | 0.827 [0.74–0.90] | 1 | 88 |
| connectivity | Random forest | subject_sgkf | 69.1 [64–73] | 67.4 [62–72] | 0.727 [0.66–0.78] | 72.7 [65–79] | 88/121 | 0.813 [0.73–0.88] | 0 | 4292 |
| raw windows | ShallowConvNet | subject_sgkf | 66.8 [62–72] | 66.2 [61–71] | 0.708 [0.64–0.77] | 74.4 [66–82] | 90/121 | 0.758 [0.66–0.84] | 3 | 6371 |
| spectral | XGBoost | subject_sgkf | 64.9 [60–70] | 64.7 [59–70] | 0.695 [0.62–0.76] | 71.1 [63–79] | 86/121 | 0.773 [0.68–0.85] | 0 | 270 |
| SMVMD | SMVMD+mRMR+KNN (nested-tuned) | subject_sgkf | 65.3 [61–70] | 64.6 [60–69] | 0.683 [0.62–0.74] | 72.7 [65–80] | 88/121 | 0.757 [0.67–0.84] | 1 | 1238 |
| spectral | Random forest | subject_sgkf | 65.6 [60–71] | 64.3 [59–70] | 0.690 [0.61–0.76] | 67.8 [60–75] | 82/121 | 0.765 [0.67–0.85] | 0 | 1367 |
| MiniRocket 10k kernels (sktime) | MiniRocket + ridge | subject_sgkf | 64.6 [60–69] | 64.1 [60–69] | 0.691 [0.63–0.75] | 75.2 [68–82] | 91/121 | 0.778 [0.69–0.86] | 0 | 1907 |
| aperiodic | Logistic regression | subject_sgkf | 63.7 [58–69] | 63.6 [58–69] | 0.672 [0.60–0.74] | 68.6 [60–77] | 83/121 | 0.721 [0.62–0.81] | 1 | 16 |
| fb_logvar | Logistic regression | subject_sgkf | 63.4 [58–69] | 63.4 [58–69] | 0.682 [0.60–0.75] | 71.9 [64–80] | 87/121 | 0.730 [0.63–0.82] | 2 | 10 |
| spectral | SVM (RBF) | subject_sgkf | 64.0 [58–70] | 63.2 [57–69] | 0.673 [0.60–0.74] | 69.4 [62–77] | 84/121 | 0.735 [0.64–0.82] | 1 | 705 |
| aperiodic | XGBoost | subject_sgkf | 62.8 [57–68] | 62.9 [57–68] | 0.677 [0.60–0.75] | 69.4 [61–78] | 84/121 | 0.737 [0.64–0.82] | 2 | 81 |
| spectral | Logistic regression | subject_sgkf | 62.3 [57–68] | 62.3 [57–68] | 0.672 [0.60–0.74] | 66.9 [59–75] | 81/121 | 0.732 [0.64–0.82] | 0 | 99 |
| wpt | Logistic regression | subject_sgkf | 61.5 [56–67] | 61.4 [56–67] | 0.664 [0.59–0.73] | 62.0 [53–71] | 75/121 | 0.719 [0.63–0.81] | 1 | 64 |
| raw windows | EEGNet-8,2 | subject_sgkf | 61.9 [57–67] | 60.2 [55–65] | 0.630 [0.56–0.69] | 62.8 [55–70] | 76/121 | 0.670 [0.58–0.76] | 1 | 3928 |
| SMVMD | SMVMD+mRMR+KNN (paper Table II) | subject_sgkf | 59.3 [56–62] | 58.9 [56–62] | 0.615 [0.57–0.65] | 64.5 [56–73] | 78/121 | 0.732 [0.64–0.81] | 4 | 4 |
| SMVMD | SMVMD+mRMR+KNN (paper Table II) | subject_gkf | 57.5 [54–61] | 57.1 [54–60] | 0.592 [0.55–0.63] | 61.2 [53–69] | 74/121 | 0.685 [0.59–0.77] | 3 | 4 |
| Euclidean-aligned covariance, tangent space | Tangent space + LR (†) | subject_sgkf | 52.0 [49–55] | 51.9 [49–55] | 0.524 [0.48–0.57] | 55.4 [46–64] | 67/121 | 0.575 [0.47–0.68] | 1 | 730 |
| Euclidean-aligned raw windows | EEGNet-8,2 (†) | subject_sgkf | 51.1 [47–55] | 50.6 [47–55] | 0.507 [0.45–0.56] | 50.4 [42–59] | 61/121 | 0.517 [0.42–0.61] | 3 | 2622 |
| spectral, z-scored within each child | Logistic regression (†) | subject_sgkf | 44.1 [41–47] | 50.0 [50–50] | 0.500 [0.50–0.50] | 49.6 [50–50] | 60/121 | 0.500 [0.50–0.50] | 0 | 9 |

(†) uses unlabelled data of the test child at test time (transductive); not comparable to strictly inductive rows.

DS-2: ADHD children were taking methylphenidate and the controls were not; ADHD cannot be separated from its treatment in this dataset.