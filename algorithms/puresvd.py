# File: algorithms/puresvd.py
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .base import Recommender

class PureSVDRecommender(Recommender):
    def __init__(self, k, **kwargs):
        super().__init__(k)
        self.name = "PureSVD"
        self.svd = TruncatedSVD(n_components=self.k, random_state=42)
        self.sigma = None

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.svd.fit(R)
        self.P = self.svd.transform(R) / self.svd.singular_values_
        self.sigma = np.diag(self.svd.singular_values_)
        self.Q = self.svd.components_.T
        singular_values = self.svd.singular_values_ # Get the values

        if visualizer:
            params_to_save = {
                'algorithm': self.name,
                'k': self.k
            }
            visualizer.visualize_fit_results(
                P=self.P,
                Q=self.Q,
                singular_values=singular_values,
                params=params_to_save
            )

        if progress_callback:
            progress_callback(1.0)
        return self