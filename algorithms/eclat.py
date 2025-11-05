import numpy as np
import pandas as pd
from itertools import combinations
from .base import Recommender

class EclatRecommender(Recommender):
    """
    Implements Eclat "from scratch" to find frequent itemsets and generate rules.
    This is an educational implementation based on the book text .
    """
    def __init__(self, k=10, min_support=0.1, min_confidence=0.5, metric='confidence', movie_titles_df=None, item_id_map=None, **kwargs):
        super().__init__(k=k)
        self.name = "Eclat"
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.metric = metric
        self.frequent_itemsets_ = pd.DataFrame()
        self.rules_ = pd.DataFrame()
        self.item_names_ = None
        self.item_map_ = None
        self.processed_rules_ = []
        self.num_users_ = 0
        self.num_items_ = 0
        self.num_transactions_ = 0
        self.movie_titles_df = movie_titles_df
        self.item_id_map = item_id_map

    def _get_frequent_itemsets(self, R_bool_sets_idx, R_bool):
        """
        Implements the Eclat algorithm using a vertical data format.
        """
        self.num_transactions_ = len(R_bool_sets_idx)

        min_support_count = self.min_support * self.num_transactions_

        # 1. Convert to Vertical TID-List (Transaction ID list) 
        # We use user indices as "Transaction IDs"
        vertical_data = {}
        for i, tid_set in enumerate(R_bool_sets_idx):
            for item_idx in tid_set:
                if item_idx not in vertical_data:
                    vertical_data[item_idx] = set()
                vertical_data[item_idx].add(i)

        # Filter 1-itemsets
        frequent_items_1 = {}
        for item_idx, tid_set in vertical_data.items():
            if len(tid_set) >= min_support_count:
                frequent_items_1[frozenset([item_idx])] = tid_set

        all_frequent_itemsets = frequent_items_1.copy()

        # 2. Recursively find frequent k-itemsets
        k = 2
        current_frequent_itemsets = frequent_items_1

        while current_frequent_itemsets:
            next_frequent_itemsets = {}
            # Join step: combine k-1 itemsets
            itemset_list = list(current_frequent_itemsets.keys())

            for i in range(len(itemset_list)):
                for j in range(i + 1, len(itemset_list)):
                    s1 = itemset_list[i]
                    s2 = itemset_list[j]

                    # Eclat join: create (k)-itemset
                    new_itemset = s1.union(s2)

                    if len(new_itemset) == k:
                        # Pruning: check if already processed
                        if new_itemset not in next_frequent_itemsets:
                            # 3. Intersect TID-Lists 
                            tid_set_1 = current_frequent_itemsets[s1]
                            tid_set_2 = current_frequent_itemsets[s2]
                            new_tid_set = tid_set_1.intersection(tid_set_2)

                            # Filter by min_support
                            if len(new_tid_set) >= min_support_count:
                                next_frequent_itemsets[new_itemset] = new_tid_set

            all_frequent_itemsets.update(next_frequent_itemsets)
            current_frequent_itemsets = next_frequent_itemsets
            k += 1

        # 4. Format as DataFrame
        itemset_data = []
        for itemset, tid_set in all_frequent_itemsets.items():
            support = len(tid_set) / self.num_transactions_
            # Convert indices back to item names
            named_itemset = frozenset([self.item_names_[idx] for idx in itemset])
            itemset_data.append({'support': support, 'itemsets': named_itemset})

        return pd.DataFrame(itemset_data)

    def _generate_rules(self, frequent_itemsets):
        """
        Generates association rules. Same logic as Apriori.
        .
        """
        rules = []
        # Create a dict for fast support lookup
        support_map = {row['itemsets']: row['support'] for _, row in frequent_itemsets.iterrows()}

        for _, row in frequent_itemsets.iterrows():
            itemset = row['itemsets']
            if len(itemset) < 2:
                continue

            support_itemset = row['support']

            for i in range(1, len(itemset)):
                for antecedents in combinations(itemset, i):
                    antecedents = frozenset(antecedents)
                    consequents = itemset - antecedents

                    support_antecedent = support_map.get(antecedents)
                    if support_antecedent is None: continue # Should not happen if itemsets are complete

                    confidence = support_itemset / support_antecedent

                    if confidence >= self.min_confidence:
                        support_consequent = support_map.get(consequents)
                        if support_consequent is None: continue

                        lift = confidence / support_consequent

                        rules.append({
                            'antecedents': antecedents,
                            'consequents': consequents,
                            'antecedent support': support_antecedent,
                            'consequent support': support_consequent,
                            'support': support_itemset,
                            'confidence': confidence,
                            'lift': lift
                        })
        return pd.DataFrame(rules)

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.train_data = R
        self.num_users_, self.num_items_ = R.shape
        self.num_transactions_ = self.num_users_
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
                print(f"Warning (Eclat): Failed to map movie titles, falling back. Error: {e}")
                self.item_names_ = [f"item_{i}" for i in range(self.num_items_)]
        else:
            # Fallback to original behavior
            self.item_names_ = [f"item_{i}" for i in range(self.num_items_)]
        self.item_map_ = {name: i for i, name in enumerate(self.item_names_)}

        # Eclat needs a list of sets of item *indices*
        R_bool_sets_idx = [set(np.where(R[u] > 0)[0]) for u in range(self.num_users_)]
        df_bool = pd.DataFrame(R > 0, columns=self.item_names_) # Not used by Eclat, but passed for consistency

        # --- Step 1: Frequent Itemset Mining ---
        self.frequent_itemsets_ = self._get_frequent_itemsets(R_bool_sets_idx, df_bool)
        if progress_callback: progress_callback(0.5)

        # --- Step 2: Rule Generation ---
        if self.frequent_itemsets_.empty:
            self.rules_ = pd.DataFrame(columns=["antecedents", "consequents", "confidence", "lift", "support"])
        else:
            self.rules_ = self._generate_rules(self.frequent_itemsets_)

        # Pre-process rules for efficient prediction
        self.processed_rules_ = []
        for row in self.rules_.itertuples():
            try:
                # Convert item names (frozenset) back to indices
                antecedents_idx = {self.item_map_[item] for item in row.antecedents}
                consequents_idx = {self.item_map_[item] for item in row.consequents}
                self.processed_rules_.append((antecedents_idx, consequents_idx, row.confidence))
            except KeyError:
                continue

        if visualizer:
            visualizer.visualize_fit_results(self.frequent_itemsets_, self.rules_, params_to_save)

        if progress_callback: progress_callback(1.0)
        return self

    # The predict() method is identical to Apriori's
    def predict(self):
        """
        Generates recommendation scores based on association rules.
        The score for an item is the highest confidence of a rule
        that recommends it, based on the user's history.
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