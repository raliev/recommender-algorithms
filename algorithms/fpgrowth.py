import numpy as np
import pandas as pd
from .base import Recommender
# This import is correct for mlxtend and does NOT include eclat
from mlxtend.frequent_patterns import fpgrowth, association_rules

class FPGrowthRecommender(Recommender):
    """
    Implements FP-Growth using the mlxtend library.
    This is a wrapper for the production-ready implementation
    cited in the book.
    """
    def __init__(self, k=10, min_support=0.1, min_confidence=0.5, metric='confidence',movie_titles_df=None, item_id_map=None, **kwargs):
        super().__init__(k=k)
        self.name = "FP-Growth"
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.metric = metric
        self.frequent_itemsets_ = None
        self.rules_ = None
        self.item_names_ = None
        self.item_map_ = None
        self.processed_rules_ = []
        self.num_users_ = 0
        self.num_items_ = 0
        self.movie_titles_df = movie_titles_df
        self.item_id_map = item_id_map

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.train_data = R
        self.num_users_, self.num_items_ = R.shape

        if self.item_id_map is not None and self.movie_titles_df is not None and len(self.item_id_map) == self.num_items_:
            try:
                # Use real titles as item names
                self.item_names_ = []
                for movie_id in self.item_id_map:
                    try:
                        self.item_names_.append(self.movie_titles_df.loc[movie_id, 'title'])
                    except KeyError:
                        self.item_names_.append(f"Movie ID {movie_id} (Unknown)")
            except Exception as e:
                print(f"Warning (FPGrowth): Failed to map movie titles, falling back. Error: {e}")
                self.item_names_ = [f"item_{i}" for i in range(self.num_items_)]
        else:
            # Fallback to original behavior
            self.item_names_ = [f"item_{i}" for i in range(self.num_items_)]

        self.item_map_ = {name: i for i, name in enumerate(self.item_names_)}

        # mlxtend requires a boolean DataFrame
        df_bool = pd.DataFrame(R > 0, columns=self.item_names_)

        # --- Step 1: Frequent Itemset Mining ---
        self.frequent_itemsets_ = fpgrowth(df_bool, min_support=self.min_support, use_colnames=True)
        if progress_callback: progress_callback(0.5)

        # --- Step 2: Rule Generation ---
        if self.frequent_itemsets_.empty:
            self.rules_ = pd.DataFrame(columns=["antecedents", "consequents", "confidence", "lift", "support"])
        else:
            self.rules_ = association_rules(
                self.frequent_itemsets_,
                metric=self.metric,
                min_threshold=self.min_confidence
            )

        # Pre-process rules for efficient prediction
        self.processed_rules_ = []
        for row in self.rules_.itertuples():
            try:
                antecedents_idx = {self.item_map_[item] for item in row.antecedents}
                consequents_idx = {self.item_map_[item] for item in row.consequents}
                self.processed_rules_.append((antecedents_idx, consequents_idx, row.confidence))
            except KeyError:
                continue

        if visualizer:
            visualizer.visualize_fit_results(self.frequent_itemsets_, self.rules_, params_to_save)

        if progress_callback: progress_callback(1.0)
        return self

    def predict(self):
        """
        Generates recommendation scores based on association rules.
        Logic based on .
        """
        R_scores = np.zeros(self.train_data.shape)

        user_item_sets = [set(np.where(self.train_data[u] > 0)[0]) for u in range(self.num_users_)]

        for u, user_set in enumerate(user_item_sets):
            if not user_set:
                continue

            scores = {}

            for antecedents_idx, consequents_idx, confidence in self.processed_rules_:
                if antecedents_idx.issubset(user_set):
                    for item_idx in consequents_idx:
                        if item_idx not in user_set:
                            scores[item_idx] = max(scores.get(item_idx, 0), confidence)

            if scores:
                item_indices = list(scores.keys())
                item_scores = list(scores.values())
                R_scores[u, item_indices] = item_scores

        self.R_predicted = R_scores
        return self.R_predicted