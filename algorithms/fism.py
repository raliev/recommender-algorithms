# app/algorithms/fism.py
import numpy as np
from .base import Recommender

class FISMRecommender(Recommender):
    """
    FISM: Factored Item Similarity Models for Top-N Recommender Systems.
    """
    def __init__(self, k, iterations=100, learning_rate=0.01, lambda_reg=0.01, alpha=0.5, **kwargs):
        super().__init__(k)
        self.name = "FISM"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.alpha = alpha  # Weight for the number of rated items
        self.P = None
        self.Q = None
        self.user_item_interactions = {}

    def fit(self, R, progress_callback=None, visualizer = None):
        num_users, num_items = R.shape

        # Initialize item latent factor matrices
        self.P = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.biases = np.zeros(num_items)

        # Pre-compute user interactions for efficiency
        for u in range(num_users):
            self.user_item_interactions[u] = R[u, :].nonzero()[0]

        if visualizer:
            params_to_save = {
                'algorithm': self.name, 'k': self.k, 'iterations_set': self.iterations,
                'learning_rate': self.learning_rate, 'lambda_reg': self.lambda_reg,
                'alpha': self.alpha
            }
            visualizer.start_run(params_to_save, R=R)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None
            for u in range(num_users):
                rated_items = self.user_item_interactions[u]
                if len(rated_items) == 0:
                    continue

                for item_i in rated_items:
                    # Calculate the sum of latent factors of items rated by the user (excluding item_i)
                    rated_items_minus_i = rated_items[rated_items != item_i]
                    num_rated_minus_i = len(rated_items_minus_i)

                    if num_rated_minus_i == 0:
                        continue

                    sum_p_j = self.P[rated_items_minus_i, :].sum(axis=0)
                    norm_factor = np.power(num_rated_minus_i, -self.alpha)

                    # Predict score
                    pred = self.biases[item_i] + (sum_p_j @ self.Q[item_i, :].T) * norm_factor
                    error = 1 - pred # For implicit feedback, target is 1

                    # Update parameters using Stochastic Gradient Descent (SGD)
                    b_i = self.biases[item_i]
                    q_i = self.Q[item_i, :]

                    self.biases[item_i] += self.learning_rate * (error - self.lambda_reg * b_i)
                    self.Q[item_i, :] += self.learning_rate * (error * norm_factor * sum_p_j - self.lambda_reg * q_i)

                    # Update P for all items j that the user has rated (excluding i)
                    for item_j in rated_items_minus_i:
                        p_j = self.P[item_j, :]
                        self.P[item_j, :] += self.learning_rate * (error * norm_factor * q_i - self.lambda_reg * p_j)
                if visualizer:
                    p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                    q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                    current_iteration = i + 1

                    visualizer.record_iteration(
                        iteration_num=current_iteration,
                        total_iterations=self.iterations,
                        P=self.P,
                        Q=self.Q,
                        p_change=p_change_norm,
                        q_change=q_change_norm,
                        # Note: FISM doesn't calculate an 'objective' here
                    )
            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        # The learned similarity matrix can be computed as P @ Q.T
        self.similarity_matrix = self.P @ self.Q.T

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        num_users, num_items = self.train_data.shape
        predictions = np.zeros((num_users, num_items))

        for u in range(num_users):
            rated_items = self.user_item_interactions[u]
            if len(rated_items) == 0:
                continue

            num_rated = len(rated_items)
            norm_factor = np.power(num_rated, -self.alpha)
            sum_p_j = self.P[rated_items, :].sum(axis=0)

            # Predict scores for all items for user u
            predictions[u, :] = self.biases + (self.Q @ sum_p_j.T) * norm_factor

        self.R_predicted = np.where(self.train_data > 0, self.train_data, predictions)
        return self.R_predicted