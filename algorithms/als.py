# algorithms/als.py
import numpy as np
from .base import Recommender
from sklearn.metrics import mean_squared_error

class ALSRecommender(Recommender):
    def __init__(self, k, iterations=10, lambda_reg=0.1):
        super().__init__(k)
        self.name = "ALS"
        self.iterations = iterations
        self.lambda_reg = lambda_reg
    def _calculate_objective(self, R, rated_mask, P, Q):
        """Calculates RMSE on the observed training ratings."""
        pred = P @ Q.T
        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        rated_mask = R > 0

        if visualizer:
            visualizer.start_run(params_to_save)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None #
            Q_old = self.Q.copy() if visualizer else None #

            for u in range(num_users): #
                rated_indices = np.where(rated_mask[u, :])[0]
                if len(rated_indices) > 0:
                    Qu = self.Q[rated_indices, :]
                    Ru = R[u, rated_indices]
                    A = Qu.T @ Qu + self.lambda_reg * np.eye(self.k) #
                    b = Qu.T @ Ru
                    self.P[u, :] = np.linalg.solve(A, b) #

            for item_idx in range(num_items): #
                rated_indices = np.where(rated_mask[:, item_idx])[0]
                if len(rated_indices) > 0:
                    Pi = self.P[rated_indices, :]
                    Ri = R[rated_indices, item_idx]
                    A = Pi.T @ Pi + self.lambda_reg * np.eye(self.k) #
                    b = Pi.T @ Ri
                    self.Q[item_idx, :] = np.linalg.solve(A, b) #

            if visualizer:
                objective = self._calculate_objective(R, rated_mask, self.P, self.Q)
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro') #
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro') #
                current_iteration = i + 1

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    objective=objective,
                    p_change=p_change_norm,
                    q_change=q_change_norm
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations) #

        if visualizer:
            visualizer.end_run()

        return self