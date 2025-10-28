import numpy as np
from .base import Recommender

class FISMRecommender(Recommender):
    """
    FISM: Factored Item Similarity Models for Top-N Recommender Systems.
    """
    def __init__(self, k, iterations=100, learning_rate=0.01, lambda_reg=0.01, alpha=0.5, **kwargs):
        super().__init__(k)
        self.name = "FISM"
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.alpha = alpha  # Weight for the number of rated items
        self.P = None
        self.Q = None
        self.biases = None # Add biases
        self.user_item_interactions = {}

    def _calculate_objective(self, R):
        """Calculates the approximate FISM objective function value."""
        num_users, num_items = R.shape
        loss = 0

        for u in range(num_users):
            rated_items = self.user_item_interactions.get(u, np.array([]))
            if len(rated_items) == 0:
                continue

            for item_i in rated_items:
                rated_items_minus_i = rated_items[rated_items != item_i]
                num_rated_minus_i = len(rated_items_minus_i)

                if num_rated_minus_i == 0:
                    pred = self.biases[item_i] # Predict using bias only if no other items
                else:
                    sum_p_j = self.P[rated_items_minus_i, :].sum(axis=0)
                    norm_factor = np.power(num_rated_minus_i, -self.alpha)
                    pred = self.biases[item_i] + (sum_p_j @ self.Q[item_i, :].T) * norm_factor

                error = 1 - pred # Target is 1 for implicit feedback
                loss += error**2

        # Add regularization
        reg_term = 0.5 * self.lambda_reg * (
                np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.biases**2)
        )
        return loss + reg_term

    def fit(self, R, progress_callback=None, visualizer = None):
        num_users, num_items = R.shape
        R_binary = (R > 0).astype(int) # Work with binary interactions

        # Initialize item latent factor matrices
        self.P = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.Q = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        self.biases = np.zeros(num_items)

        # Pre-compute user interactions for efficiency
        self.user_item_interactions.clear() # Clear previous interactions if any
        for u in range(num_users):
            self.user_item_interactions[u] = R_binary[u, :].nonzero()[0] # Use R_binary

        if visualizer:
            params_to_save = {
                'algorithm': self.name, 'k': self.k, 'iterations_set': self.iterations,
                'learning_rate': self.learning_rate, 'lambda_reg': self.lambda_reg,
                'alpha': self.alpha
            }
            visualizer.start_run(params_to_save, R=R_binary) # Pass binary R

        for i in range(self.iterations):
            P_old = self.P.copy() if visualizer else None
            Q_old = self.Q.copy() if visualizer else None
            for u in range(num_users):
                rated_items = self.user_item_interactions[u]
                if len(rated_items) == 0:
                    continue

                for item_i in rated_items:
                    # Calculate the sum of latent factors of items rated by the user (excluding item_i)
                    rated_items_minus_i = rated_items[rated_items != item_i]
                    num_rated_minus_i = len(rated_items_minus_i)

                    if num_rated_minus_i == 0:
                        sum_p_j = np.zeros(self.k) # No other items, sum is zero
                        norm_factor = 0 # Avoid division by zero, prediction relies only on bias
                    else:
                        sum_p_j = self.P[rated_items_minus_i, :].sum(axis=0)
                        norm_factor = np.power(num_rated_minus_i, -self.alpha)

                    # Predict score
                    pred = self.biases[item_i] + (sum_p_j @ self.Q[item_i, :].T) * norm_factor
                    error = 1 - pred # For implicit feedback, target is 1

                    # Update parameters using Stochastic Gradient Descent (SGD)
                    b_i = self.biases[item_i]
                    q_i = self.Q[item_i, :]

                    self.biases[item_i] += self.learning_rate * (error - self.lambda_reg * b_i)
                    self.Q[item_i, :] += self.learning_rate * (error * norm_factor * sum_p_j - self.lambda_reg * q_i)

                    # Update P for all items j that the user has rated (excluding i)
                    if num_rated_minus_i > 0: # Only update if there were other items
                        grad_p_common = error * norm_factor * q_i
                        self.P[rated_items_minus_i, :] += self.learning_rate * (
                                grad_p_common - self.lambda_reg * self.P[rated_items_minus_i, :]
                        )

            # Moved visualizer block outside inner loops for efficiency
            if visualizer:
                p_change_norm = np.linalg.norm(self.P - P_old, 'fro')
                q_change_norm = np.linalg.norm(self.Q - Q_old, 'fro')
                current_iteration = i + 1
                objective = self._calculate_objective(R_binary) # Calculate objective

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.iterations,
                    P=self.P,
                    Q=self.Q,
                    objective=objective, # Pass objective
                    p_change=p_change_norm,
                    q_change=q_change_norm,
                )

            if progress_callback:
                progress_callback((i + 1) / self.iterations)

        # The learned similarity matrix can be computed as P @ Q.T
        self.similarity_matrix = self.P @ self.Q.T

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        num_users, num_items = self.train_data.shape # Use train_data shape
        predictions = np.zeros((num_users, num_items))

        for u in range(num_users):
            rated_items = self.user_item_interactions.get(u, np.array([])) # Use precomputed list
            if len(rated_items) == 0:
                # Predict global average or zero if no history? Let's use bias average.
                # predictions[u, :] = self.biases.mean() # Simplification
                continue # Or leave as zeros

            num_rated = len(rated_items)

            # --- FISM Prediction Logic ---
            # Predict score for item i: b_i + (sum_{j in R_u \setminus \{i\}} p_j) * q_i / |R_u \setminus \{i\}|^alpha
            # This is complex to vectorize efficiently. Let's approximate with R_u @ (P @ Q^T) for speed,
            # as done in the visualizer's breakdown plot. The 'fit' uses the correct logic.
            # OR iterate for precise prediction (might be slow):
            sum_p_all_rated = self.P[rated_items, :].sum(axis=0)

            for item_i in range(num_items):
                # We need to predict for *all* items, including those already rated,
                # then filter later if needed.
                if item_i in rated_items:
                    # Calculate sum excluding item_i
                    rated_items_minus_i = rated_items[rated_items != item_i]
                    num_rated_minus_i = len(rated_items_minus_i)
                    if num_rated_minus_i == 0:
                        sum_p_j = np.zeros(self.k)
                        norm_factor = 0
                    else:
                        sum_p_j = self.P[rated_items_minus_i, :].sum(axis=0)
                        norm_factor = np.power(num_rated_minus_i, -self.alpha)
                else:
                    # Predicting for an unrated item, use all rated items in context
                    sum_p_j = sum_p_all_rated
                    num_rated_all = len(rated_items)
                    if num_rated_all == 0:
                        norm_factor = 0
                    else:
                        norm_factor = np.power(num_rated_all, -self.alpha)

                predictions[u, item_i] = self.biases[item_i] + (sum_p_j @ self.Q[item_i, :].T) * norm_factor

        # Store prediction but don't overwrite training data (implicit model)
        self.R_predicted = predictions
        return self.R_predicted