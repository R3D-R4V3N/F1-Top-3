import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.base import clone

from .time_series import GroupTimeSeriesSplit


def find_best_threshold(model, X, y, groups, metric="pr"):
    """Estimate the optimal probability cutoff using cross-validated predictions.

    Parameters
    ----------
    model : estimator object
        The estimator pipeline. ``predict_proba`` must be implemented.
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix used for fitting the estimator.
    y : array-like
        True target labels.
    groups : array-like
        Group labels for the ``GroupTimeSeriesSplit``.
    metric : {"pr", "roc"}, optional
        If ``"pr"`` (default), the threshold with the highest F1-score on the
        precision-recall curve is returned. If ``"roc"``, the threshold that
        maximises TPR - FPR on the ROC curve is returned.

    Returns
    -------
    float
        The selected probability threshold.
    """

    cv = GroupTimeSeriesSplit(n_splits=5)
    proba = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y, groups):
        clf = clone(model)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        clf.fit(X_train, y_train)
        proba[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    if metric == "roc":
        fpr, tpr, thresholds = roc_curve(y, proba)
        j_scores = tpr - fpr
        best_idx = int(np.nanargmax(j_scores))
        return float(thresholds[best_idx])

    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])
