import numpy as np
from .base import Recommender

class BPRAdaptiveRecommender(Recommender):
    """
    Implements BPR with Adaptive Negative Sampling.
    Instead of uniform negative sampling, it samples a small pool
    of random negatives and picks the one with the highest current
    predicted score ("hard negative") for the gradient update.
    """
    def __init__(self, k, iterations=1000, learning_rate=0.01, lambda_reg=0.01, negative_sample_pool_size=5):
        super().__init__(k)
        self.name = "BPR (Adaptive)" # User-facing name
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.negative_sample_pool_size = negative_sample_pool_size

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

            if visualizer:
                epoch_negative_scores = [] # To track the "hardness"

            np.random.shuffle(positive_pairs)

            for u, item_i in positive_pairs:

                # Part 3: Adaptive Negative Sampling
                best_j = -1
                # We want the j with the highest score (hardest negative)
                best_j_score = -np.inf

                for _ in range(self.negative_sample_pool_size):
                    # Sample a random item 'j'
                    j = np.random.randint(num_items)
                    # Keep sampling until we find a true negative
                    while R_binary[u, j] > 0:
                        j = np.random.randint(num_items)

                    # Get the score for this potential negative item
                    score_j = self.P[u, :] @ self.Q[j, :].T

                    # If it's the hardest one we've seen, keep it
                    if score_j > best_j_score:
                        best_j_score = score_j
                        best_j = j

                j = best_j # Our chosen hard negative

                if visualizer:
                    epoch_negative_scores.append(best_j_score)

                # We now have our triplet (u, i, j)

                # Part 4: Gradient Calculation and Update
                p_u, q_i, q_j = self.P[u, :], self.Q[item_i, :], self.Q[j, :]
                x_uij = p_u @ q_i.T - p_u @ q_j.T
                sigmoid_x = 1 / (1 + np.exp(x_uij))

                self.P[u, :] += self.learning_rate * (sigmoid_x * (q_i - q_j) - self.lambda_reg * p_u)
                self.Q[item_i, :] += self.learning_rate * (sigmoid_x * p_u - self.lambda_reg * q_i)
                self.Q[j, :] += self.learning_rate * (sigmoid_x * -p_u - self.lambda_reg * q_j)

            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                current_iteration = i + 1
                avg_neg_score = np.mean(epoch_negative_scores) if epoch_negative_scores else 0

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    p_change=p_change_norm,
                    q_change=q_change_norm,
                    avg_negative_score=avg_neg_score # Pass new metric
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        # After fitting, P and Q are learned
        if self.P is not None and self.Q is not None:
            return self.P @ self.Q.T
        else:
            raise RuntimeError("The model has not been fitted yet.")