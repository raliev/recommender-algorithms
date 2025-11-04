# algorithms/base.py
class Recommender:
    """
    Base class for all recommendation models.
    """
    def __init__(self, k, **kwargs):
        self.k = k
        self.P = None
        self.Q = None
        self.R_predicted = None
        self.name = "BaseRecommender"

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        """
        Trains the model on the ratings matrix R.
        """
        raise NotImplementedError

    def predict(self):
        """
        Generates the matrix of predicted ratings.
        """
        self.R_predicted = self.P @ self.Q.T
        return self.R_predicted