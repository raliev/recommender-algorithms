import numpy as np
from .base import Recommender

class WRMFRecommender(Recommender):
    def __init__(self, k, iterations=10, lambda_reg=0.1, alpha=40):
        super().__init__(k)
        self.name = "WRMF"
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.alpha = alpha

    def _calculate_objective_for_visualizer(self, R, C, P, Q):
        P_u, Q_i = R.nonzero()
        p_ui = 1
        C_ui = C[P_u, Q_i]
        predictions_interactions = np.sum(P[P_u, :] * Q[Q_i, :], axis=1)
        squared_error_interactions = (p_ui - predictions_interactions) ** 2
        weighted_error_interactions = np.sum(C_ui * squared_error_interactions)
        reg_term = self.lambda_reg * (np.sum(np.linalg.norm(P, axis=1)**2) + np.sum(np.linalg.norm(Q, axis=1)**2))
        return weighted_error_interactions + reg_term

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        num_users, num_items = R.shape
        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        if visualizer:
            visualizer.start_run(params_to_save, R=R)

        C = 1 + self.alpha * R
        p_ui_matrix = (R > 0).astype(int)

        for i in range(self.iterations):
            P_old = self.P.copy()
            Q_old = self.Q.copy()

            for u in range(num_users):
                Cu_diag = C[u, :]
                Cu = np.diag(Cu_diag)
                A = self.Q.T @ Cu @ self.Q + self.lambda_reg * np.eye(self.k)
                b = self.Q.T @ Cu @ p_ui_matrix[u, :]
                self.P[u, :] = np.linalg.solve(A, b)

            for item_idx in range(num_items):
                Ci_diag = C[:, item_idx]
                Ci = np.diag(Ci_diag)
                A = self.P.T @ Ci @ self.P + self.lambda_reg * np.eye(self.k)
                b = self.P.T @ Ci @ p_ui_matrix[:, item_idx]
                self.Q[item_idx, :] = np.linalg.solve(A, b)

            current_iteration = i + 1

            if visualizer:
                objective_value = self._calculate_objective_for_visualizer(R, C, self.P, self.Q)
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations, # Pass total iterations
                    P=self.P,
                    Q=self.Q,
                    objective=objective_value,
                    p_change=p_change_norm,
                    q_change=q_change_norm
                )

            if progress_callback:
                progress_callback(current_iteration / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        self.R_predicted = self.P @ self.Q.T
        return self.R_predicted