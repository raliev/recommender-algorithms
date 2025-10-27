# algorithms/bpr.py
import numpy as np
from .base import Recommender

class BPRRecommender(Recommender):
    def __init__(self, k, iterations=1000, learning_rate=0.01, lambda_reg=0.01):
        super().__init__(k)
        self.name = "BPR"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def fit(self, R, progress_callback=None, visualizer = None):
        R_binary = (R > 0).astype(int)
        num_users, num_items = R_binary.shape
        self.P = np.random.normal(scale=0.1, size=(num_users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(num_items, self.k))
        positive_pairs = np.argwhere(R_binary > 0)

        if visualizer:
            params_to_save = {
                'algorithm': self.name, 'k': self.k,
                'iterations_set': self.iterations,
                'learning_rate': self.learning_rate,
                'lambda_reg': self.lambda_reg
            }
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
                    q_change=q_change_norm
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self