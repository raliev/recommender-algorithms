import numpy as np
from .base import Recommender

class EASERecommender(Recommender):
    """
    Implements Embarrassingly Shallow Autoencoders (EASE) for Top-N
    recommendation from implicit feedback.

    This model learns a linear item-item similarity matrix B by solving
    a closed-form regression problem.

    """
    def __init__(self, k=10, lambda_reg=100.0, **kwargs):
        # k is not used for training, but is passed to the visualizer
        # for the recommendation breakdown plot.
        super().__init__(k)
        self.name = "EASE"
        self.lambda_reg = lambda_reg
        self.similarity_matrix = None # This will be the matrix B

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        """
        Computes the EASE item-item similarity matrix B using the
        closed-form analytical solution.

        """
        self.train_data = R

        if progress_callback:
            progress_callback(0.1)

        # 1. Calculate the Gram matrix (item-item co-occurrence)
        # G = X^T * X
        G = self.train_data.T @ self.train_data

        if progress_callback:
            progress_callback(0.3)

        # 2. Add L2 regularization (lambda * I)
        G_reg = G + self.lambda_reg * np.eye(G.shape[0])

        if progress_callback:
            progress_callback(0.5)

        # 3. Calculate the intermediate matrix P
        # P = (G + lambda*I)^-1
        #
        try:
            P = np.linalg.inv(G_reg)
        except np.linalg.LinAlgError:
            print("Error: Gram matrix is singular, cannot invert. Try increasing lambda_reg.")
            # Use pseudo-inverse as a fallback
            P = np.linalg.pinv(G_reg)

        if progress_callback:
            progress_callback(0.8)

        # 4. Calculate the final similarity matrix B
        # B_ij = -P_ij / P_jj (for i != j)
        # B_ii = 0
        P_diag = np.diag(P)
        B = -P / P_diag # This divides P_ij by P_jj (element-wise broadcasting)

        # Set diagonal to zero
        np.fill_diagonal(B, 0)

        self.similarity_matrix = B

        if visualizer:
            visualizer.visualize_fit_results(
                B=self.similarity_matrix,
                R=R,
                params=params_to_save
            )

        if progress_callback:
            progress_callback(1.0)

        return self

    def predict(self):
        """
        Generates prediction scores by multiplying the original
        interaction matrix X with the learned similarity matrix B.

        """
        if self.similarity_matrix is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")

        # R_predicted = X * B
        self.R_predicted = self.train_data @ self.similarity_matrix

        return self.R_predicted