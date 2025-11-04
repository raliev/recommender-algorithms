import numpy as np
from .base import Recommender
from sklearn.metrics import mean_squared_error # Import RMSE

class SVDppRecommender(Recommender):
    """
    Implements the full SVD++ algorithm, including implicit feedback.
    Prediction Model:
    r_ui = μ + b_u + b_i + (p_u + |N(u)|^(-1/2) * Σ(y_j for j in N(u)))^T * q_i
    """
    def __init__(self, k=2, iterations=20, learning_rate=0.005,
                 lambda_p=0.02, lambda_q=0.02, lambda_y=0.02,
                 lambda_bu=0.02, lambda_bi=0.02):
        self.k = k
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.name = "SVD++"
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.lambda_y = lambda_y
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

        self.P = None # user factors (explicit)
        self.Q = None # item factors (for prediction)
        self.Y = None # item factors (for implicit user profile)

        self._user_rated_items = None
        self._user_norm_factors = None
        self.rated_mask = None # Add this
        self.R_train = None # Add this

    def _calculate_objective(self, R, rated_mask):
        """Calculates RMSE on the observed training ratings."""
        # Need to reconstruct predictions using the full SVD++ formula
        num_users, num_items = R.shape
        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], num_items, axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], num_users, axis=0)

        full_user_factors = np.copy(self.P)
        for u in range(num_users):
            rated_items = self._user_rated_items[u]
            norm_factor = self._user_norm_factors[u]
            if rated_items.size > 0:
                implicit_factor_sum = self.Y[rated_items, :].sum(axis=0)
                full_user_factors[u, :] += norm_factor * implicit_factor_sum

        pred = self.global_mean + user_bias_matrix + item_bias_matrix + (full_user_factors @ self.Q.T)

        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        if observed_actuals.size == 0:
            return 0.0
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))


    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        num_users, num_items = R.shape
        self.R_train = R # Store R
        self.rated_mask = R > 0 # Store mask

        self.P = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.Y = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = R[R > 0].mean() if np.any(R>0) else 0 # Handle empty R

        self._user_rated_items = [
            np.where(R[u, :] > 0)[0] for u in range(num_users)
        ]
        self._user_norm_factors = np.zeros(num_users)
        for u in range(num_users):
            item_count = len(self._user_rated_items[u])
            if item_count > 0:
                self._user_norm_factors[u] = 1.0 / np.sqrt(item_count)

        if visualizer:
            visualizer.start_run(params_to_save)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None
            Y_old = self.Y.copy() if visualizer else None

            for u, item_idx in np.argwhere(R > 0):
                rated_items = self._user_rated_items[u]
                norm_factor = self._user_norm_factors[u]
                implicit_factor_sum = np.zeros(self.k) # Initialize to zero
                if rated_items.size > 0:
                    implicit_factor_sum = self.Y[rated_items, :].sum(axis=0) # Sum if items exist
                p_u_full = self.P[u, :] + norm_factor * implicit_factor_sum

                pred = self.global_mean + self.user_bias[u] + self.item_bias[item_idx] + p_u_full @ self.Q[item_idx, :].T
                error = R[u, item_idx] - pred

                bu = self.user_bias[u]
                bi = self.item_bias[item_idx]
                pu = self.P[u, :]
                qi = self.Q[item_idx, :]

                # Update biases
                self.user_bias[u] += self.learning_rate * (error - self.lambda_bu * bu)
                self.item_bias[item_idx] += self.learning_rate * (error - self.lambda_bi * bi)

                # Update explicit user factor P
                self.P[u, :] += self.learning_rate * (error * qi - self.lambda_p * pu)

                # Update item factor Q
                self.Q[item_idx, :] += self.learning_rate * (error * p_u_full - self.lambda_q * qi)

                # Update ALL implicit item factors Y in N(u)
                if rated_items.size > 0: # Only update if user rated items
                    grad_y_common = error * norm_factor * qi
                    self.Y[rated_items, :] += self.learning_rate * (
                            grad_y_common - self.lambda_y * self.Y[rated_items, :]
                    )

            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                y_change_norm = np.linalg.norm(self.Y - Y_old, 'fro')
                current_iteration = i + 1
                objective = self._calculate_objective(self.R_train, self.rated_mask) # Calculate RMSE

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    Y=self.Y,
                    objective=objective, # Pass objective
                    p_change=p_change_norm,
                    q_change=q_change_norm,
                    y_change=y_change_norm
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        if self.P is None or self.Q is None or self.Y is None:
            raise RuntimeError("Model must be trained (call .fit()) before predicting.")

        num_users, num_items = self.P.shape[0], self.Q.shape[0]

        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], num_items, axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], num_users, axis=0)

        full_user_factors = np.copy(self.P)
        for u in range(num_users):
            rated_items = self._user_rated_items[u]
            norm_factor = self._user_norm_factors[u]

            if rated_items.size > 0:
                implicit_factor_sum = self.Y[rated_items, :].sum(axis=0)
                full_user_factors[u, :] += norm_factor * implicit_factor_sum

        self.R_predicted = self.global_mean + user_bias_matrix + item_bias_matrix + (full_user_factors @ self.Q.T)

        return self.R_predicted