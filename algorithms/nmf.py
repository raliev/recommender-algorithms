# algorithms/nmf.py
import numpy as np
from sklearn.decomposition import NMF
from .base import Recommender

class NMFRecommender(Recommender):
    def __init__(self, k, max_iter=200, **kwargs):
        super().__init__(k)
        self.name = "NMF"
        self.model = NMF(
            n_components=self.k,
            init='random',
            random_state=42,
            max_iter=max_iter,
            l1_ratio=0, # L2 regularization
            alpha_W=0.01, # Regularization on P
            alpha_H=0.01  # Regularization on Q
        )

    def fit(self, R, progress_callback=None, visualizer = None):
        # NMF requires non-negative inputs
        R_non_negative = np.maximum(R, 0)
        self.P = self.model.fit_transform(R_non_negative)
        self.Q = self.model.components_.T

        if progress_callback:
            progress_callback(1.0)
        return self

    def predict(self):
        # The base predict method is correct for NMF
        return super().predict()