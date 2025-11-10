import numpy as np
from .base import Recommender

try:
    from sklearn.linear_model import ElasticNet
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SLIMRecommender(Recommender):
    def __init__(self, k=0, l1_reg=0.001, l2_reg=0.0001, **kwargs):
        # k is not used by SLIM, but kept for compatibility
        super().__init__(k)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed. Please run 'pip install scikit-learn' to use SLIM.")

        self.name = "SLIM"
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.similarity_matrix = None

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.train_data = R
        num_items = R.shape[1]
        self.similarity_matrix = np.zeros((num_items, num_items))

        # We use ElasticNet which combines L1 and L2 penalties.
        # alpha corresponds to the overall regularization strength
        # l1_ratio corresponds to the mix between L1 and L2
        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha if alpha > 0 else 0

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=True, # Enforce non-negative coefficients for interpretability
            fit_intercept=False,
            copy_X=False
        )

        # Ignore convergence warnings for a cleaner demo output
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Train a model for each item
        for j in range(num_items):
            # Target item j
            y = R[:, j]

            # All other items are features
            X = np.delete(R, j, axis=1)

            # Fit the model
            model.fit(X, y)

            # Insert the learned coefficients back into the similarity matrix
            self.similarity_matrix[j, :] = np.insert(model.coef_, j, 0)

            if progress_callback:
                progress_callback((j + 1) / num_items)

        warnings.resetwarnings()

        if visualizer:
            params_to_save = {
                'algorithm': self.name,
                'l1_reg': self.l1_reg,
                'l2_reg': self.l2_reg
            }
        visualizer.visualize_fit_results(
            W=self.similarity_matrix,
            R=R,
            params=params_to_save)

        return self

    def predict(self):
        # Prediction is a matrix multiplication of user ratings and the learned similarity matrix
        self.R_predicted = self.train_data @ self.similarity_matrix
        return np.where(self.train_data > 0, self.train_data, self.R_predicted)