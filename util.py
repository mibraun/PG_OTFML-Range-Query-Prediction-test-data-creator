import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.externals.six import string_types
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)


class FittingNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'x_min_'):
            del self.x_min_
            del self.x_max_

    def fit(self, X, y=None):
        self._reset()
        self.x_min_ = np.asarray(np.amin(X, axis=0))
        self.x_max_ = np.asarray(np.amax(X, axis=0))
        return self

    def transform(self, X, y='deprecated', copy=None):
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        check_is_fitted(self, 'x_min_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                if self.x_max_[col] == self.x_min_[col]:
                    X[row, col] = 0
                else:
                    X[row, col] = (X[row, col] - self.x_min_[col]) / (self.x_max_[col] - self.x_min_[col])
        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self, 'x_min_')

        copy = copy if copy is not None else self.copy
        if copy:
            X = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j] = (X[i, j] * (self.x_max_[i] - self.x_min_[i]) + self.x_min_[i])
        return X
