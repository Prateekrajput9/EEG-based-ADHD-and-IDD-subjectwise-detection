"""Classical classifiers and their (small) nested-CV grids.

Every pipeline starts with a StandardScaler, so scaling is refitted on each
training fold. Class imbalance (DS-2: 61 ADHD / 60 NC children but 3680 / 2908
windows) is handled with balanced class weights.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost with per-fold balanced sample weights (computed from training labels only)."""

    def __init__(self, max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.5, seed=42):
        self.max_depth = max_depth; self.n_estimators = n_estimators
        self.learning_rate = learning_rate; self.subsample = subsample
        self.colsample_bytree = colsample_bytree; self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier
        y = np.asarray(y).astype(int)
        w = (len(y) / (2 * np.bincount(y, minlength=2).clip(1)))[y]
        self.model_ = XGBClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                                    learning_rate=self.learning_rate, subsample=self.subsample,
                                    colsample_bytree=self.colsample_bytree, tree_method='hist',
                                    n_jobs=1, random_state=self.seed, eval_metric='logloss')
        self.model_.fit(X, y, sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make(name: str, seed: int = 42):
    """Return (pipeline, grid). Grids are deliberately small: few children, few folds."""
    if name == 'lr':
        clf = LogisticRegression(class_weight='balanced', max_iter=3000)
        grid = {'clf__C': [0.01, 0.1, 1.0]}
    elif name == 'svm':
        clf = SVC(kernel='rbf', class_weight='balanced', gamma='scale')
        grid = {'clf__C': [0.1, 1.0, 10.0]}
    elif name == 'rf':
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                     max_features='sqrt', n_jobs=1, random_state=seed)
        grid = {'clf__max_depth': [6, None], 'clf__min_samples_leaf': [1, 10]}
    elif name == 'xgb':
        clf = BalancedXGB(seed=seed)
        grid = {'clf__max_depth': [2, 4]}
    else:
        raise ValueError(name)
    return Pipeline([('scale', StandardScaler()), ('clf', clf)]), grid


NAMES = {'lr': 'Logistic regression', 'svm': 'SVM (RBF)', 'rf': 'Random forest', 'xgb': 'XGBoost'}
