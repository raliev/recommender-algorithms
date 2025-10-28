import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemKNNRecommender:
    """
    An Item-based k-Nearest Neighbors recommender.

    This recommender predicts user ratings for items based on the ratings
    of similar items that the user has already rated.

    Parameters
    ----------
    k : int, optional
        The number of nearest neighbors to consider for prediction, by default 2.
    min_support : int, optional
        The minimum number of co-raters required for a similarity score
        to be considered, by default 1.
    shrinkage : float, optional
        The shrinkage factor to apply to similarity scores to discount
        those with low support, by default 2.0.
    """
    def __init__(self, k=2, min_support=1, shrinkage=2.0):
        self.k = k
        self.min_support = min_support
        self.shrinkage = shrinkage
        self.similarity_matrix_ = None
        self.train_data_ = None
        self.user_means_ = None

    def fit(self, R):
        """
        Trains the model by computing the item-item similarity matrix.

        Parameters
        ----------
        R : np.ndarray
            The user-item interaction matrix of shape (num_users, num_items).

        Returns
        -------
        self
            The fitted recommender instance.
        """
        self.train_data_ = R
        num_items = R.shape[1]
        self.similarity_matrix_ = np.zeros((num_items, num_items))

        # Calculate the mean rating for each user who has rated at least one item
        self.user_means_ = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
        # Replace NaN (from users with no ratings) with 0
        self.user_means_[np.isnan(self.user_means_)] = 0

        # Center the ratings by subtracting the user's mean
        R_centered = R - np.where(R != 0, self.user_means_[:, np.newaxis], 0)

        # Calculate cosine similarity on the centered matrix
        self.similarity_matrix_ = cosine_similarity(R_centered.T)

        # Calculate co-rated counts
        co_rated_counts = (self.train_data_ > 0).astype(float).T @ (self.train_data_ > 0).astype(float)

        # Apply min_support threshold
        if self.min_support > 0:
            self.similarity_matrix_[co_rated_counts < self.min_support] = 0

        # Apply shrinkage factor
        if self.shrinkage > 0:
            shrinkage_factor = co_rated_counts / (co_rated_counts + self.shrinkage)
            self.similarity_matrix_ *= shrinkage_factor

        # An item cannot be its own neighbor
        np.fill_diagonal(self.similarity_matrix_, 0)

        return self

    def predict_for_user(self, user_ratings):
        """
        Predicts ratings for all unrated items for a single user.

        Parameters
        ----------
        user_ratings : np.ndarray
            A 1D array representing a single user's ratings.

        Returns
        -------
        np.ndarray
            A 1D array with predicted scores for the user's unrated items.
        """
        predictions = np.zeros_like(user_ratings, dtype=float)

        # Find items the user has already rated
        rated_indices = np.where(user_ratings > 0)[0]

        # Iterate over all items to predict scores for unrated ones
        for item_to_predict in range(len(user_ratings)):
            if user_ratings[item_to_predict] == 0:
                # Get similarities between the target item and all items the user has rated
                sims_to_rated_items = self.similarity_matrix_[item_to_predict, rated_indices]

                # Get the user's ratings for those items
                ratings_of_rated_items = user_ratings[rated_indices]

                # Select the top K nearest neighbors
                if self.k > 0 and len(sims_to_rated_items) > self.k:
                    # Find indices of the top-k absolute similarities
                    top_neighbors_indices = np.argsort(-np.abs(sims_to_rated_items))[:self.k]

                    # Filter down to only the top-k neighbors
                    sims_to_rated_items = sims_to_rated_items[top_neighbors_indices]
                    ratings_of_rated_items = ratings_of_rated_items[top_neighbors_indices]

                # Calculate the weighted average
                numerator = sims_to_rated_items @ ratings_of_rated_items
                denominator = np.sum(np.abs(sims_to_rated_items))

                if denominator > 1e-8: # Add a small epsilon to avoid division by zero
                    predictions[item_to_predict] = numerator / denominator

        return predictions

if __name__ == '__main__':
    # Create a sample user-item rating matrix
    # Rows: Users, Columns: Items
    # 0 indicates no rating
    R = np.array([
        [5, 3, 0, 1, 4, 0],
        [4, 0, 0, 1, 3, 5],
        [1, 1, 5, 5, 0, 4],
        [0, 1, 4, 4, 0, 0],
        [2, 0, 3, 0, 5, 4],
    ])

    # 1. Instantiate the recommender with hyperparameters
    item_knn = ItemKNNRecommender(k=2, min_support=1, shrinkage=1.0)

    # 2. Train the model on the rating data
    item_knn.fit(R)

    # 3. Get recommendations for a specific user (e.g., user 0)
    target_user_ratings = R[0, :]
    predicted_scores = item_knn.predict_for_user(target_user_ratings)

    # 4. Generate the final ranked list
    # Create a list of (item_id, score) for unrated items
    unrated_items = np.where(target_user_ratings == 0)[0]
    recommendations = sorted(
        zip(unrated_items, predicted_scores[unrated_items]),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Original ratings for User 0: {target_user_ratings}")
    print(f"Predicted scores for User 0: {np.round(predicted_scores, 2)}")
    print("\nTop recommendations for User 0:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: Predicted Score = {score:.4f}")