import numpy as np
from .base import Recommender

class BPRSVDPPRecommender(Recommender):
    def __init__(self, k, iterations=1000, learning_rate=0.01,
                 lambda_p=0.01, lambda_q=0.01, lambda_y=0.01):

        super().__init__(k)
        self.name = "BPR+SVDPP"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.lambda_y = lambda_y

        self.P = None  # User latent factors (Explicit)
        self.Q = None  # Item latent factors (For prediction)
        self.Y = None  # Item latent factors (For implicit profile)

        self._user_rated_items = None
        self._user_norm_factors = None

    def _calculate_train_auc(self, R_binary, positive_pairs, num_samples=10000):
        """
        Calculates the AUC on the training data.
        A true validation AUC would use a separate, held-out set of triplets.
        """
        num_items = R_binary.shape[1]
        if len(positive_pairs) == 0:
            return 0.5  # No positive pairs, return random guess score

        # Sample from the positive pairs
        sampled_indices = np.random.choice(len(positive_pairs), num_samples, replace=True)
        sampled_pairs = positive_pairs[sampled_indices]

        correct_predictions = 0
        for u, i in sampled_pairs:
            # Sample a negative item
            j = np.random.randint(num_items)
            while R_binary[u, j] > 0:
                j = np.random.randint(num_items)

            # Get scores
            score_i = self.P[u, :] @ self.Q[i, :].T
            score_j = self.P[u, :] @ self.Q[j, :].T

            if score_i > score_j:
                correct_predictions += 1

        return correct_predictions / num_samples

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        R_binary = (R > 0).astype(int)
        num_users, num_items = R_binary.shape
        self.P = np.random.normal(scale=0.1, size=(num_users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(num_items, self.k))
        self.Y = np.random.normal(scale=0.1, size=(num_items, self.k))
        positive_pairs = np.argwhere(R_binary > 0)
        self._user_rated_items = [
            np.where(R_binary[u, :] > 0)[0] for u in range(num_users)
        ]
        self._user_norm_factors = np.zeros(num_users)
        for u in range(num_users):
            item_count = len(self._user_rated_items[u])
            if item_count > 0:
                self._user_norm_factors[u] = 1.0 / np.sqrt(item_count)
        if visualizer:
            visualizer.start_run(params_to_save, R=R_binary)

        for i in range(self.iterations):

            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None

            np.random.shuffle(positive_pairs)
            for u, item_i in positive_pairs:
                j = np.random.randint(num_items)
                while R_binary[u, j] > 0:
                    j = np.random.randint(num_items)

                p_u = self.P[u, :]
                q_i = self.Q[item_i, :]
                q_j = self.Q[j, :]

                rated_items = self._user_rated_items[u]
                norm_factor = self._user_norm_factors[u]

                # 2. Construct the full user profile
                implicit_factor_sum = np.zeros(self.k)
                if rated_items.size > 0:
                    implicit_factor_sum = self.Y[rated_items, :].sum(axis=0)

                p_u_full = p_u + norm_factor * implicit_factor_sum

                # 3. Calculate score difference using the *full* profile
                x_uij = (p_u_full @ q_i.T) - (p_u_full @ q_j.T) # <-- KEY CHANGE

                sigmoid_x = 1 / (1 + np.exp(x_uij))

                # 4. Calculate gradients
                grad_common = self.learning_rate * sigmoid_x
                grad_q_diff = grad_common * (q_i - q_j)
                grad_pu_common = grad_common * p_u_full

                # 5. Update all parameters
                self.P[u, :] += grad_q_diff - (self.learning_rate * self.lambda_p * p_u)
                self.Q[item_i, :] += grad_pu_common - (self.learning_rate * self.lambda_q * q_i)
                self.Q[j, :] += -grad_pu_common - (self.learning_rate * self.lambda_q * q_j)

                # --- NEW: Update all implicit Y factors ---
                if rated_items.size > 0:
                    grad_y_common = grad_q_diff * norm_factor
                    self.Y[rated_items, :] += grad_y_common - (self.learning_rate * self.lambda_y * self.Y[rated_items, :])

            if visualizer or progress_callback:
                current_iteration = i + 1
                auc_score = 0.0

                if visualizer:
                    p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                    q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')

                    auc_score = self._calculate_train_auc(R_binary, positive_pairs, num_samples=10000)

                    visualizer.record_iteration(
                        iteration_num=current_iteration,
                        total_iterations=self.iterations,
                        P=self.P,
                        Q=self.Q,
                        p_change=p_change_norm,
                        q_change=q_change_norm,
                        auc=auc_score  # <-- Pass the new metric
                    )

                if progress_callback:
                    progress_callback(current_iteration / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        if self.P is None or self.Q is None or self.Y is None: # <-- MODIFIED
            raise RuntimeError("Model must be trained (call .fit()) before predicting.")

        num_users, num_items = self.P.shape[0], self.Q.shape[0]

        full_user_factors = np.copy(self.P)
        for u in range(num_users):
            rated_items = self._user_rated_items[u]
            norm_factor = self._user_norm_factors[u]
            if rated_items.size > 0:
                implicit_factor_sum = self.Y[rated_items, :].sum(axis=0)
                full_user_factors[u, :] += norm_factor * implicit_factor_sum

        self.R_predicted = full_user_factors @ self.Q.T
        return self.R_predicted