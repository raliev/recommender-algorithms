# algorithms/asvd.py
import numpy as np
from .base import Recommender
from sklearn.metrics import mean_squared_error

class ASVDRecommender(Recommender):
    """
    Implements Asymmetric SVD (ASVD).
    This model removes the explicit user factor P_u and replaces it
    with a profile constructed from the items the user has interacted with,
    using both explicit (x_j) and implicit (y_j) item factors.

    Prediction Model:
    r_ui = μ + b_u + b_i + q_i^T * (
        |R(u)|^(-1/2) * Σ_{j in R(u)} (r_uj - b_uj) * x_j +
        |N(u)|^(-1/2) * Σ_{j in N(u)} y_j
    )
    """
    def __init__(self, k, iterations=20, learning_rate=0.005,
                 lambda_q=0.02, lambda_x=0.02, lambda_y=0.02,
                 lambda_bu=0.02, lambda_bi=0.02):
        super().__init__(k)
        self.name = "ASVD"
        self.iterations = iterations
        self.learning_rate = learning_rate

        # Regularization terms
        self.lambda_q = lambda_q
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi

        # Biases
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

        # Item Factors (No User Factors)
        self.Q = None # item factors (for prediction)
        self.X = None # item factors (for explicit history)
        self.Y = None # item factors (for implicit history)

        # Caching for user history
        self._user_rated_items = None
        self._user_implicit_items = None
        self._user_norm_explicit = None
        self._user_norm_implicit = None

        self.rated_mask = None
        self.R_train = None

    def _get_baseline(self, u, i):
        """Calculates the baseline bias b_ui = μ + b_u + b_i"""
        return self.global_mean + self.user_bias[u] + self.item_bias[i]

    def _get_user_profile(self, u):
        """Constructs the composite user profile vector for user u."""
        rated_items = self._user_rated_items[u]
        implicit_items = self._user_implicit_items[u]

        explicit_sum = np.zeros(self.k)
        if rated_items.size > 0:
            baselines = self._get_baseline(u, rated_items)
            residuals = self.R_train[u, rated_items] - baselines
            explicit_sum = residuals @ self.X[rated_items, :]

        implicit_sum = np.zeros(self.k)
        if implicit_items.size > 0:
            implicit_sum = self.Y[implicit_items, :].sum(axis=0)

        return (self._user_norm_explicit[u] * explicit_sum +
                self._user_norm_implicit[u] * implicit_sum)

    def _calculate_objective(self, R, rated_mask):
        """Calculates RMSE on the observed training ratings."""
        num_users, num_items = R.shape
        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], num_items, axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], num_users, axis=0)

        # Construct full user profile matrix (slow, but accurate for objective)
        full_user_profiles = np.zeros((num_users, self.k))
        for u in range(num_users):
            full_user_profiles[u, :] = self._get_user_profile(u)

        pred = (self.global_mean + user_bias_matrix + item_bias_matrix +
                (full_user_profiles @ self.Q.T))

        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        if observed_actuals.size == 0:
            return 0.0
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        num_users, num_items = R.shape
        self.R_train = R
        self.rated_mask = R > 0

        # Initialize factor matrices (all item-based)
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.X = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.Y = np.random.normal(scale=1./self.k, size=(num_items, self.k))

        # Initialize biases
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = R[R > 0].mean() if np.any(R > 0) else 0

        # Pre-cache user interaction lists and normalization factors
        self._user_rated_items = [R[u, :].nonzero()[0] for u in range(num_users)]
        self._user_implicit_items = self._user_rated_items # For ASVD, R(u) == N(u)

        self._user_norm_explicit = np.zeros(num_users)
        self._user_norm_implicit = np.zeros(num_users)
        for u in range(num_users):
            explicit_count = len(self._user_rated_items[u])
            if explicit_count > 0:
                self._user_norm_explicit[u] = 1.0 / np.sqrt(explicit_count)

            implicit_count = len(self._user_implicit_items[u])
            if implicit_count > 0:
                self._user_norm_implicit[u] = 1.0 / np.sqrt(implicit_count)

        if visualizer:
            visualizer.start_run(params_to_save)

        for i in range(self.iterations):
            X_old = self.X.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None
            Y_old = self.Y.copy() if visualizer else None

            for u, item_idx in np.argwhere(R > 0):
                # Get user profile and baseline
                user_profile = self._get_user_profile(u)
                baseline = self._get_baseline(u, item_idx)

                # Predict and calculate error
                pred = baseline + user_profile @ self.Q[item_idx, :].T
                error = R[u, item_idx] - pred

                # Get terms for updates
                bu = self.user_bias[u]
                bi = self.item_bias[item_idx]
                qi = self.Q[item_idx, :]

                # Update biases
                self.user_bias[u] += self.learning_rate * (error - self.lambda_bu * bu)
                self.item_bias[item_idx] += self.learning_rate * (error - self.lambda_bi * bi)

                # Update Q_i (Item factor)
                self.Q[item_idx, :] += self.learning_rate * (error * user_profile - self.lambda_q * qi)

                # Common gradients for X and Y
                grad_common = error * qi

                # Update X_j for all j in R(u) (explicit)
                rated_items = self._user_rated_items[u]
                if rated_items.size > 0:
                    norm_exp = self._user_norm_explicit[u]
                    residuals = (self.R_train[u, rated_items] -
                                 self._get_baseline(u, rated_items))
                    grad_x_common = grad_common * norm_exp

                    # This is the tricky part: grad_x_j = grad_common * (r_uj - b_uj)
                    # We can't easily vectorize the (r_uj - b_uj) * x_j part in the grad
                    # Let's use an outer product
                    grad_X = np.outer(residuals, grad_x_common)

                    self.X[rated_items, :] += self.learning_rate * (
                            grad_X - self.lambda_x * self.X[rated_items, :]
                    )

                # Update Y_j for all j in N(u) (implicit)
                implicit_items = self._user_implicit_items[u]
                if implicit_items.size > 0:
                    norm_imp = self._user_norm_implicit[u]
                    grad_y_common = grad_common * norm_imp
                    self.Y[implicit_items, :] += self.learning_rate * (
                            grad_y_common - self.lambda_y * self.Y[implicit_items, :]
                    )

            if visualizer:
                x_change_norm = np.linalg.norm(self.X - X_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                y_change_norm = np.linalg.norm(self.Y - Y_old, 'fro')
                current_iteration = i + 1
                objective = self._calculate_objective(self.R_train, self.rated_mask)

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    X=self.X,
                    Q=self.Q,
                    Y=self.Y,
                    objective=objective,
                    x_change=x_change_norm,
                    q_change=q_change_norm,
                    y_change=y_change_norm
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        if self.Q is None or self.X is None or self.Y is None:
            raise RuntimeError("Model must be trained (call .fit()) before predicting.")

        num_users, num_items = self.R_train.shape

        user_bias_matrix = np.repeat(self.user_bias[:, np.newaxis], num_items, axis=1)
        item_bias_matrix = np.repeat(self.item_bias[np.newaxis, :], num_users, axis=0)

        # Construct full user profile matrix
        full_user_profiles = np.zeros((num_users, self.k))
        for u in range(num_users):
            full_user_profiles[u, :] = self._get_user_profile(u)

        self.R_predicted = (self.global_mean + user_bias_matrix + item_bias_matrix +
                            (full_user_profiles @ self.Q.T))

        return self.R_predicted