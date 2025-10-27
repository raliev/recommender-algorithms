import numpy as np
from .base import Recommender
from sklearn.metrics.pairwise import cosine_similarity

class ItemKNNRecommender(Recommender):
    def __init__(self, k=20, similarity_metric='cosine', min_support=2, shrinkage=0.0, **kwargs):
        super().__init__(k)
        self.name = "ItemKNN"
        self.k = k # Number of neighbors
        self.similarity_matrix = None
        self.train_data = None
        self.similarity_metric = similarity_metric
        self.min_support = min_support
        self.shrinkage = shrinkage

    def fit(self, R, progress_callback=None, visualizer=None):
        self.train_data = R
        num_items = R.shape[1]
        raw_similarity_matrix = np.zeros((num_items, num_items), dtype=float)

        if self.similarity_metric == 'cosine':
            raw_similarity_matrix = cosine_similarity(R.T)
        elif self.similarity_metric == 'adjusted_cosine':
            user_means = R.mean(axis=1)
            user_means[np.isnan(user_means)] = 0 # Handle users with no ratings
            R_centered = R - np.where(R > 0, user_means[:, np.newaxis], 0) # Use R > 0
            raw_similarity_matrix = cosine_similarity(R_centered.T)
        elif self.similarity_metric == 'pearson':
            raw_similarity_matrix = np.corrcoef(R.T)
            raw_similarity_matrix = np.nan_to_num(raw_similarity_matrix)

        self.similarity_matrix = raw_similarity_matrix.copy()

        co_rated_counts = (self.train_data > 0).astype(float).T @ (self.train_data > 0).astype(float)

        if self.min_support > 0:
            self.similarity_matrix[co_rated_counts < self.min_support] = 0

        if self.shrinkage > 0:
            self.similarity_matrix = (co_rated_counts / (co_rated_counts + self.shrinkage)) * self.similarity_matrix

        # We don't want an item to be its own neighbor, so we set diagonal to 0
        np.fill_diagonal(self.similarity_matrix, 0)

        if visualizer:
            params_to_save = {
                'algorithm': self.name,
                'k_neighbors': self.k,
                'similarity_metric': self.similarity_metric,
                'min_support': self.min_support,
                'shrinkage': self.shrinkage
            }
            visualizer.visualize_fit_results(
                R=R,
                raw_similarity_matrix=raw_similarity_matrix,
                final_similarity_matrix=self.similarity_matrix,
                co_rated_counts_matrix=co_rated_counts,
                params=params_to_save
            )
        if progress_callback:
            progress_callback(1.0)
        return self

    def predict_for_user(self, user_ratings):
        predictions = np.zeros_like(user_ratings, dtype=float)

        for item_to_predict in range(len(user_ratings)):
            # Only predict for items the user hasn't rated
            if user_ratings[item_to_predict] == 0:

                rated_indices = np.where(user_ratings > 0)[0]

                # The similarity scores of the item to predict to the items the user has rated
                sims_to_rated_items = self.similarity_matrix[item_to_predict, rated_indices]

                # The ratings the user gave to those items
                ratings_of_rated_items = user_ratings[rated_indices]

                # Get top K neighbors
                if self.k > 0 and len(sims_to_rated_items) > self.k:
                    top_neighbors_indices = np.argsort(-np.abs(sims_to_rated_items))[:self.k]
                    sims_to_rated_items = sims_to_rated_items[top_neighbors_indices]
                    ratings_of_rated_items = ratings_of_rated_items[top_neighbors_indices]

                # Weighted sum of similarities
                numerator = sims_to_rated_items @ ratings_of_rated_items
                denominator = np.abs(sims_to_rated_items).sum() + 1e-8

                if denominator > 0:
                    predictions[item_to_predict] = numerator / denominator
        return predictions

    def predict(self):
        num_users, _ = self.train_data.shape
        predictions = np.zeros_like(self.train_data, dtype=float)

        for u in range(num_users):
            predictions[u, :] = self.predict_for_user(self.train_data[u, :])

        self.R_predicted = np.where(self.train_data > 0, self.train_data, predictions)
        return self.R_predicted