import numpy as np
from .base import Recommender
from sklearn.metrics import mean_squared_error

class NMFRecommender(Recommender):
    def __init__(self, k, iterations=50, learning_rate=0.005, lambda_reg=0.02, **kwargs):
        super().__init__(k)
        self.name = "NMF"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.rated_mask = None
        self.R_train = None

    def _calculate_objective(self, R, rated_mask, P, Q):
        pred = P @ Q.T
        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        if observed_actuals.size == 0:
            return 0.0
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        num_users, num_items = R.shape
        self.R_train = R
        self.rated_mask = R > 0

        self.P = np.random.uniform(0, 1./self.k, size=(num_users, self.k))
        self.Q = np.random.uniform(0, 1./self.k, size=(num_items, self.k))

        if visualizer:
            visualizer.start_run(params_to_save, R=self.R_train)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None

            for u, item_idx in np.argwhere(self.rated_mask):
                error = R[u, item_idx] - self.P[u, :] @ self.Q[item_idx, :].T

                p_u = self.P[u, :]
                q_i = self.Q[item_idx, :]

                grad_p = (error * q_i - self.lambda_reg * p_u)
                grad_q = (error * p_u - self.lambda_reg * q_i)

                self.P[u, :] += self.learning_rate * grad_p
                self.Q[item_idx, :] += self.learning_rate * grad_q


                self.P[u, :] = np.maximum(self.P[u, :], 0)
                self.Q[item_idx, :] = np.maximum(self.Q[item_idx, :], 0)

            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                current_iteration = i + 1

                objective = self._calculate_objective(self.R_train, self.rated_mask, self.P, self.Q)

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
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        return super().predict()