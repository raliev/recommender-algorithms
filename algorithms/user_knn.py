# app/algorithms/user_knn.py
import numpy as np
from .base import Recommender
from sklearn.metrics.pairwise import cosine_similarity

class UserKNNRecommender(Recommender):
    def __init__(self, k=20, similarity_metric='cosine', **kwargs):
        super().__init__(k)
        self.name = "UserKNN"
        self.k = k # Number of neighbors
        self.similarity_metric = similarity_metric
        self.similarity_matrix = None
        self.train_data = None
        self.user_means = None

    def fit(self, R, progress_callback=None, visualizer = None):
        self.train_data = R

        num_items = R.shape[1]
        raw_similarity_matrix = np.zeros((num_items, num_items), dtype=float)

        if self.similarity_metric == 'cosine':
            raw_similarity_matrix = cosine_similarity(self.train_data)
        elif self.similarity_metric == 'adjusted_cosine' or self.similarity_metric == 'pearson':
            # For user-based CF, Pearson and Adjusted Cosine are effectively the same.
            # Center ratings by subtracting the user's mean for each rating.
            self.user_means = np.true_divide(self.train_data.sum(axis=1), (self.train_data != 0).sum(axis=1))
            self.user_means[np.isnan(self.user_means)] = 0
            R_centered = self.train_data - np.where(self.train_data != 0, self.user_means[:, np.newaxis], 0)
            raw_similarity_matrix = cosine_similarity(R_centered)

        self.similarity_matrix = raw_similarity_matrix.copy()

        # A user cannot be their own neighbor
        np.fill_diagonal(self.similarity_matrix, 0)
        if visualizer:
            params_to_save = {
                'algorithm': self.name,
                'k_neighbors': self.k,
                'similarity_metric': self.similarity_metric
            }
            visualizer.visualize_fit_results(
                raw_similarity_matrix=raw_similarity_matrix,
                final_similarity_matrix=self.similarity_matrix,
                params=params_to_save
            )
        if progress_callback:
            progress_callback(1.0)
        return self

    def predict(self):
        num_users, num_items = self.train_data.shape
        predictions = np.zeros_like(self.train_data, dtype=float)

        # Pre-calculate user means if not already done (for cosine similarity)
        if self.user_means is None:
            self.user_means = np.true_divide(self.train_data.sum(axis=1), (self.train_data != 0).sum(axis=1))
            self.user_means[np.isnan(self.user_means)] = 0

        for u in range(num_users):
            similar_users_indices = np.argsort(-self.similarity_matrix[u, :])

            for i in range(num_items):
                # Predict only for unrated items
                if self.train_data[u, i] == 0:
                    numerator = 0
                    denominator = 0

                    neighbor_count = 0
                    for neighbor_idx in similar_users_indices:
                        if neighbor_count >= self.k:
                            break

                        # Check if the neighbor has rated this item
                        if self.train_data[neighbor_idx, i] > 0:
                            sim = self.similarity_matrix[u, neighbor_idx]
                            neighbor_rating = self.train_data[neighbor_idx, i]
                            neighbor_mean = self.user_means[neighbor_idx]

                            numerator += sim * (neighbor_rating - neighbor_mean)
                            denominator += np.abs(sim)
                            neighbor_count += 1

                    if denominator > 0:
                        pred = self.user_means[u] + (numerator / denominator)
                        predictions[u, i] = np.clip(pred, 0, 5) # Clamp to valid rating range
                    else:
                        predictions[u, i] = self.user_means[u] # Fallback to user mean

        self.R_predicted = np.where(self.train_data > 0, self.train_data, predictions)
        return self.R_predicted