from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomScaler(BaseEstimator,TransformerMixin):
    # note: returns the feature matrix with the binary columns ordered first
    def __init__(self,bin_vars_index,cont_vars_index,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.bin_vars_index = bin_vars_index
        self.cont_vars_index = cont_vars_index

    def fit(self, X, y=None):
        self.scaler.fit(X[:,self.cont_vars_index], y)
        return self

    def transform(self, X, y=None, copy=None):
        X_tail = self.scaler.transform(X[:,self.cont_vars_index],y,copy)
        return np.concatenate((X_tail, X[:,self.bin_vars_index]), axis=1)