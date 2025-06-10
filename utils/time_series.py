"""Utility functions and classes for time series cross-validation."""

import numpy as np

try:
    from sklearn.model_selection import GroupTimeSeriesSplit
except ImportError:  # scikit-learn < 1.3
    class GroupTimeSeriesSplit:
        """Simple backport that keeps complete groups in each split."""

        def __init__(self, n_splits: int = 5):
            self.n_splits = n_splits

        def split(self, X, y=None, groups=None):
            if groups is None:
                raise ValueError("The 'groups' parameter is required")
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            test_size = n_groups // (self.n_splits + 1)
            for i in range(self.n_splits):
                train_end = test_size * (i + 1)
                test_end = test_size * (i + 2)
                train_groups = unique_groups[:train_end]
                test_groups = unique_groups[train_end:test_end]
                train_idx = np.where(np.isin(groups, train_groups))[0]
                test_idx = np.where(np.isin(groups, test_groups))[0]
                yield train_idx, test_idx

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

__all__ = ["GroupTimeSeriesSplit"]
