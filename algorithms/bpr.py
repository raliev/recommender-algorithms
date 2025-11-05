import numpy as np
from .base import Recommender

class BPRRecommender(Recommender):
    def __init__(self, k, iterations=1000, learning_rate=0.01, lambda_reg=0.01):

        super().__init__(k)
        self.name = "BPR"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

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
        positive_pairs = np.argwhere(R_binary > 0)

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

                p_u, q_i, q_j = self.P[u, :], self.Q[item_i, :], self.Q[j, :]
                x_uij = p_u @ q_i.T - p_u @ q_j.T
                sigmoid_x = 1 / (1 + np.exp(x_uij))

                self.P[u, :] += self.learning_rate * (sigmoid_x * (q_i - q_j) - self.lambda_reg * p_u)
                self.Q[item_i, :] += self.learning_rate * (sigmoid_x * p_u - self.lambda_reg * q_i)
                self.Q[j, :] += self.learning_rate * (sigmoid_x * -p_u - self.lambda_reg * q_j)

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
        self.R_predicted = self.P @ self.Q.T
        return self.R_predicted