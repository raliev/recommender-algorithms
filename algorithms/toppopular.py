import numpy as np
from .base import Recommender

class TopPopularRecommender(Recommender):
    """
    Recommends items based on their overall popularity across all users.
    Popularity is measured by the total number of interactions.
    This implementation is based on the book text.
    """
    def __init__(self,movie_titles_df=None, item_id_map=None,**kwargs):
        # k is not used for training, but k=0 is a placeholder
        super().__init__(k=0)
        self.name = "Top Popular"
        self.item_popularity_ = None
        self.num_users_ = 0
        self.num_items_ = 0
        self.movie_titles_df = movie_titles_df
        self.item_id_map = item_id_map

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        """
        Calculates the popularity of each item.
        Popularity is the sum of interactions for each item (column).
        """
        self.train_data = R
        self.num_users_, self.num_items_ = R.shape

        # Calculate popularity as the sum of interactions for each item (axis=0)
        #
        self.item_popularity_ = np.array(R.sum(axis=0)).flatten()

        if visualizer:
            if hasattr(visualizer, 'set_title_maps'):
                visualizer.set_title_maps(self.movie_titles_df, self.item_id_map)
            visualizer.visualize_fit_results(self.item_popularity_, params_to_save)

        if progress_callback:
            progress_callback(1.0)

        return self

    def predict(self):
        """
        Generates a prediction matrix where every user's score for an item
        is that item's global popularity.
        """
        if self.item_popularity_ is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")

        # Create a matrix by tiling the popularity vector for each user
        R_scores = np.tile(self.item_popularity_, (self.num_users_, 1))

        # We set already-interacted items to -inf so they won't be recommended
        self.R_predicted = np.where(self.train_data > 0, -np.inf, R_scores)

        return self.R_predicted