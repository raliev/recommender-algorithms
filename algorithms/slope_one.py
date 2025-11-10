import numpy as np
from .base import Recommender

class SlopeOneRecommender(Recommender):
    def __init__(self, k=20, **kwargs):
        # k is not used, but kept for compatibility
        super().__init__(k)
        self.name = "Slope One"
        self.dev_matrix = None 
        self.freq_matrix = None
        self.train_data = None

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.train_data = R
        num_items = R.shape[1]
        self.dev_matrix = np.zeros((num_items, num_items), dtype=float)
        self.freq_matrix = np.zeros((num_items, num_items), dtype=int)

        for u_ratings in R:
            rated_indices = np.where(u_ratings > 0)[0] 
            # For every pair of items the user has rated
            for i in rated_indices:
                for j in rated_indices:
                    if i != j:
                        self.dev_matrix[i, j] += u_ratings[i] - u_ratings[j] 
                        self.freq_matrix[i, j] += 1

        # Pre-compute average deviations
        self.dev_matrix = np.where(
            self.freq_matrix > 0,
            self.dev_matrix / self.freq_matrix,
            0
            ) 

        if visualizer:
            params_to_save = {
                'algorithm': self.name,
            }
            visualizer.visualize_fit_results(
                dev_matrix=self.dev_matrix,
                freq_matrix=self.freq_matrix,
                params=params_to_save
            )

        if progress_callback:
            progress_callback(1.0)
        return self

    def predict(self):
        num_users, num_items = self.train_data.shape
        predictions = np.zeros_like(self.train_data, dtype=float)

        for u in range(num_users):
            user_ratings = self.train_data[u, :]
            rated_indices = np.where(user_ratings > 0)[0] 

            for j in range(num_items):
                # Predict only for items the user hasn't rated
                if user_ratings[j] == 0:
                    numerator = 0
                    denominator = 0 
                    for i in rated_indices:
                        if self.freq_matrix[j, i] > 0:
                            numerator += (user_ratings[i] + self.dev_matrix[j, i]) * self.freq_matrix[j, i]
                            denominator += self.freq_matrix[j, i] 

                    if denominator > 0:
                        predictions[u, j] = numerator / denominator

        self.R_predicted = np.where(self.train_data > 0, self.train_data, predictions)
        return self.R_predicted