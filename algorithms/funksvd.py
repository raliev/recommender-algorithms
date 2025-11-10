import numpy as np
from .base import Recommender
from sklearn.metrics import mean_squared_error

class FunkSVDRecommender(Recommender):
    def __init__(self, k, iterations=100, learning_rate=0.005, lambda_reg=0.02):
        super().__init__(k)
        self.name = "FunkSVD"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.rated_mask = None # Add this
        self.R_train = None # Add this

    def _calculate_objective(self, R, rated_mask, P, Q):
        """Calculates RMSE on the observed training ratings."""
        pred = P @ Q.T
        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        if observed_actuals.size == 0:
            return 0.0
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        num_users, num_items = R.shape
        self.R_train = R # Store R
        self.rated_mask = R > 0 # Store the mask
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        if visualizer:
            # Pass R to start_run for the breakdown plot
            visualizer.start_run(params_to_save, R=self.R_train)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None

            for u, item_idx in np.argwhere(R > 0):
                error = R[u, item_idx] - self.P[u, :] @ self.Q[item_idx, :].T
                p_u = self.P[u, :]
                q_i = self.Q[item_idx, :]

                self.P[u, :] += self.learning_rate * (error * q_i - self.lambda_reg * p_u)
                self.Q[item_idx, :] += self.learning_rate * (error * p_u - self.lambda_reg * q_i)

            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                current_iteration = i + 1

                # Calculate training RMSE
                objective = self._calculate_objective(self.R_train, self.rated_mask, self.P, self.Q)

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    objective=objective, # Pass the new objective
                    p_change=p_change_norm,
                    q_change=q_change_norm
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            R_predicted_final = self.P @ self.Q.T
            visualizer.end_run(R_predicted_final=R_predicted_final)

        return self