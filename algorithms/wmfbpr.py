import numpy as np
from .base import Recommender
import streamlit as st

class WMFBPRRecommender(Recommender):
    """
    Implements Weighted Matrix Factorization with Bayesian Personalized Ranking (WMFBPR).
    This model pre-calculates item importance (w_i) using PageRank on a
    co-occurrence graph and integrates this weight into the BPR optimization.

    Score: r_ui = P_u * (Q_i + w_i)
    """
    def __init__(self, k, iterations=1000, learning_rate=0.01, lambda_reg=0.01, pagerank_alpha=0.85):
        super().__init__(k)
        self.name = "WMFBPR"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.pagerank_alpha = pagerank_alpha
        self.item_weights = None # w(i)

    def _calculate_pagerank(self, R_binary, max_iter=100, tol=1.0e-6):
        """Calculates PageRank scores for items based on co-occurrence."""
        num_items = R_binary.shape[1]

        # 1. Item Co-occurrence (CR)
        # CR_ij = number of users who interacted with both i and j
        CR = R_binary.T @ R_binary
        np.fill_diagonal(CR, 0) # Remove self-links for PageRank

        # 2. Item Correlation (CM) - row-normalized transition matrix
        row_sums = CR.sum(axis=1)
        # Handle divide-by-zero for items with no co-occurrences
        row_sums[row_sums == 0] = 1
        CM = CR / row_sums[:, np.newaxis]

        # 3. PageRank (PR) - iterative power method
        pr = np.ones(num_items) / num_items
        damping_factor = (1 - self.pagerank_alpha) / num_items

        for _ in range(max_iter):
            pr_new = damping_factor + self.pagerank_alpha * (CM.T @ pr)

            # Check for convergence
            if np.linalg.norm(pr_new - pr, 1) < tol:
                break
            pr = pr_new

        # 4. Normalization (w(i))
        if pr.max() == pr.min():
            return np.zeros(num_items) # All items are equally (un)important

        weights = (pr - pr.min()) / (pr.max() - pr.min())
        return weights

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        R_binary = (R > 0).astype(int)
        num_users, num_items = R_binary.shape

        self.P = np.random.normal(scale=0.1, size=(num_users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(num_items, self.k))

        with st.spinner("Calculating item weights (PageRank)..."):
            self.item_weights = self._calculate_pagerank(R_binary)

        positive_pairs = np.argwhere(R_binary > 0)

        if visualizer:
            # Pass the weights to the visualizer
            visualizer.start_run(params_to_save, R=R_binary, weights=self.item_weights)

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None

            np.random.shuffle(positive_pairs)
            for u, item_i in positive_pairs:
                # Sample negative item
                j = np.random.randint(num_items)
                while R_binary[u, j] > 0:
                    j = np.random.randint(num_items)

                # Get vectors and weights
                p_u = self.P[u, :]
                q_i = self.Q[item_i, :]
                q_j = self.Q[j, :]
                w_i = self.item_weights[item_i]
                w_j = self.item_weights[j]

                boosted_q_i = q_i + w_i
                boosted_q_j = q_j + w_j

                r_uij = p_u @ boosted_q_i - p_u @ boosted_q_j

                sigmoid_x = 1 / (1 + np.exp(r_uij))

                grad_common = sigmoid_x

                # User update
                self.P[u, :] += self.learning_rate * (
                        grad_common * (boosted_q_i - boosted_q_j) - self.lambda_reg * p_u
                )

                # Positive item update
                self.Q[item_i, :] += self.learning_rate * (
                        grad_common * p_u - self.lambda_reg * boosted_q_i
                )

                # Negative item update
                self.Q[j, :] += self.learning_rate * (
                        grad_common * (-p_u) - self.lambda_reg * boosted_q_j
                )

            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                current_iteration = i + 1

                # Calculate AUC (reusing BPR's logic)
                auc_score = self._calculate_train_auc(R_binary, positive_pairs, num_samples=10000)

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    p_change=p_change_norm,
                    q_change=q_change_norm,
                    auc=auc_score
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        if visualizer:
            visualizer.end_run()

        return self

    def _calculate_train_auc(self, R_binary, positive_pairs, num_samples=10000):
        """
        Calculates the AUC on the training data using the WMFBPR score.
        """
        num_items = R_binary.shape[1]
        if len(positive_pairs) == 0:
            return 0.5

        sampled_indices = np.random.choice(len(positive_pairs), num_samples, replace=True)
        sampled_pairs = positive_pairs[sampled_indices]

        correct_predictions = 0
        for u, i in sampled_pairs:
            # Sample a negative item
            j = np.random.randint(num_items)
            while R_binary[u, j] > 0:
                j = np.random.randint(num_items)

            # Get weighted scores
            score_i = self.P[u, :] @ (self.Q[i, :] + self.item_weights[i])
            score_j = self.P[u, :] @ (self.Q[j, :] + self.item_weights[j])

            if score_i > score_j:
                correct_predictions += 1

        return correct_predictions / num_samples

    def predict(self):
        """
        Predicts scores using the weighted formula: R_ui = P_u * (Q_i + w_i)
        """
        # Create a matrix of weights, tiled for each user
        # item_weights is (n_items,), we need (n_items, k)
        # No, the weight is added to the vector, so (n_items, k) + (n_items, 1) isn't right
        # The formula is sum(P_uf * (Q_if + w_i))
        # This is P @ (Q + w_vector_tiled).T

        # Let's create the boosted Q matrix: Q_boosted = Q + w
        # self.item_weights is (n_items,). We need to add it to each factor k.
        # So we tile it.
        w_tiled = np.tile(self.item_weights[:, np.newaxis], (1, self.k))
        Q_boosted = self.Q + w_tiled

        self.R_predicted = self.P @ Q_boosted.T
        return self.R_predicted