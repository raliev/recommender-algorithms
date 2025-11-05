import numpy as np
import pandas as pd
from itertools import combinations
from .base import Recommender

class AprioriRecommender(Recommender):
    """
    Implements Apriori "from scratch" to find frequent itemsets and generate rules.
    This is an educational implementation based on the book text.
    """
    def __init__(self, k=10, min_support=0.1, min_confidence=0.5, metric='confidence', **kwargs):
        super().__init__(k=k)
        self.name = "Apriori"
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.metric = metric # Note: We will use confidence for rule generation
        self.frequent_itemsets_ = pd.DataFrame()
        self.rules_ = pd.DataFrame()
        self.item_names_ = None
        self.item_map_ = None
        self.processed_rules_ = []
        self.num_users_ = 0
        self.num_items_ = 0
        self.num_transactions_ = 0

    def _find_frequent_1_itemsets(self, R_bool):
        """Finds frequent itemsets of size 1."""
        item_counts = R_bool.sum(axis=0)
        support = item_counts / self.num_transactions_
        frequent_1_itemsets = support[support >= self.min_support]

        # Format for mlxtend-style DataFrame
        df = frequent_1_itemsets.to_frame(name='support')
        df['itemsets'] = [frozenset([col]) for col in df.index]
        return df[['support', 'itemsets']]

    def _join_and_prune(self, L_k_minus_1, k):
        """
        Generates candidate k-itemsets (C_k) from frequent (k-1)-itemsets.
        Follows the Apriori "Join Step" and "Prune Step".
        """
        candidates = set()
        prev_itemsets = list(L_k_minus_1['itemsets'])

        # Join Step
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                s1 = prev_itemsets[i]
                s2 = prev_itemsets[j]

                # Join if they share k-2 items
                if len(s1.union(s2)) == k:
                    candidates.add(s1.union(s2))

        # Prune Step
        pruned_candidates = set()
        for candidate in candidates:
            is_valid = True
            # Check all k-1 subsets of the candidate
            for subset in combinations(candidate, k - 1):
                if frozenset(subset) not in prev_itemsets:
                    is_valid = False # Apriori Principle
                    break
            if is_valid:
                pruned_candidates.add(candidate)

        return pruned_candidates

    def _get_frequent_itemsets(self, R_bool):
        """
        Implements the level-wise Apriori algorithm.
        """
        all_frequent_itemsets = []

        # Level 1 (k=1)
        L_k = self._find_frequent_1_itemsets(R_bool)
        all_frequent_itemsets.append(L_k)

        k = 2
        while not L_k.empty:
            # Generate candidates C_k
            candidates_k = self._join_and_prune(L_k, k)
            if not candidates_k:
                break

            # Count support for candidates
            support_counts = {itemset: 0 for itemset in candidates_k}
            for transaction in R_bool:
                transaction_set = set(transaction)
                for itemset in candidates_k:
                    if itemset.issubset(transaction_set):
                        support_counts[itemset] += 1

            # Filter for min_support
            L_k_data = []
            for itemset, count in support_counts.items():
                support = count / self.num_transactions_
                if support >= self.min_support:
                    L_k_data.append({'support': support, 'itemsets': itemset})

            if not L_k_data:
                break

            L_k = pd.DataFrame(L_k_data)
            all_frequent_itemsets.append(L_k)
            k += 1

        if not all_frequent_itemsets:
            return pd.DataFrame(columns=['support', 'itemsets'])

        return pd.concat(all_frequent_itemsets, ignore_index=True)

    def _generate_rules(self, frequent_itemsets):
        """
        Generates association rules based on confidence.
        """
        rules = []
        for _, row in frequent_itemsets.iterrows():
            itemset = row['itemsets']
            if len(itemset) < 2:
                continue

            support_itemset = row['support']

            for i in range(1, len(itemset)):
                for antecedents in combinations(itemset, i):
                    antecedents = frozenset(antecedents)
                    consequents = itemset - antecedents

                    # Get support for the antecedent
                    support_antecedent = frequent_itemsets[
                        frequent_itemsets['itemsets'] == antecedents
                        ]['support'].values[0]

                    confidence = support_itemset / support_antecedent

                    if confidence >= self.min_confidence:
                        support_consequent = frequent_itemsets[
                            frequent_itemsets['itemsets'] == consequents
                            ]['support'].values[0]

                        lift = confidence / support_consequent #

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
        self.item_names_ = [f"item_{i}" for i in range(self.num_items_)]
        self.item_map_ = {name: i for i, name in enumerate(self.item_names_)}

        # We need the itemsets as sets of item indices for counting
        # And also the boolean DataFrame for `_find_frequent_1_itemsets`
        R_bool_sets = [set(np.where(R[u] > 0)[0]) for u in range(self.num_users_)]
        df_bool = pd.DataFrame(R > 0, columns=self.item_names_)

        # --- Step 1: Frequent Itemset Mining ---
        self.frequent_itemsets_ = self._get_frequent_itemsets(R_bool_sets, df_bool)
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

    # --- This is a fix for the _get_frequent_itemsets method signature ---
    # We redefine `fit` and `_get_frequent_itemsets` to pass the correct data types

    def _find_frequent_1_itemsets(self, df_bool):
        """Finds frequent itemsets of size 1."""
        item_counts = df_bool.sum(axis=0)
        support = item_counts / self.num_transactions_
        frequent_1_itemsets_series = support[support >= self.min_support]

        # Format for mlxtend-style DataFrame
        df = frequent_1_itemsets_series.to_frame(name='support')
        df['itemsets'] = [frozenset([col]) for col in df.index]
        return df[['support', 'itemsets']], set(frequent_1_itemsets_series.index)

    def _get_frequent_itemsets(self, R_bool_sets, df_bool):
        """
        Implements the level-wise Apriori algorithm.
        """
        all_frequent_itemsets_dfs = []

        # Level 1 (k=1)
        L_k_df, L_k_itemsets_flat = self._find_frequent_1_itemsets(df_bool)
        all_frequent_itemsets_dfs.append(L_k_df)

        k = 2
        while not L_k_df.empty:
            # Generate candidates C_k
            candidates_k = self._join_and_prune(L_k_df, k)
            if not candidates_k:
                break

            # Count support for candidates
            support_counts = {itemset: 0 for itemset in candidates_k}
            for transaction_set in R_bool_sets:
                for itemset in candidates_k:
                    # Convert item names to indices for issubset check
                    itemset_indices = {self.item_map_[item] for item in itemset}
                    if itemset_indices.issubset(transaction_set):
                        support_counts[itemset] += 1

            # Filter for min_support
            L_k_data = []
            new_L_k_itemsets_flat = set()
            for itemset, count in support_counts.items():
                support = count / self.num_transactions_
                if support >= self.min_support:
                    L_k_data.append({'support': support, 'itemsets': itemset})
                    new_L_k_itemsets_flat.update(itemset)

            if not L_k_data:
                break

            L_k_df = pd.DataFrame(L_k_data)
            all_frequent_itemsets_dfs.append(L_k_df)
            k += 1

        if not all_frequent_itemsets_dfs:
            return pd.DataFrame(columns=['support', 'itemsets'])

        return pd.concat(all_frequent_itemsets_dfs, ignore_index=True)

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.train_data = R
        self.num_users_, self.num_items_ = R.shape
        self.num_transactions_ = self.num_users_
        self.item_names_ = [f"item_{i}" for i in range(self.num_items_)]
        self.item_map_ = {name: i for i, name in enumerate(self.item_names_)}

        # We need both the boolean DataFrame (for L1) and a list of sets (for L_k counting)
        df_bool = pd.DataFrame(R > 0, columns=self.item_names_)
        R_bool_sets = [set(np.where(R[u] > 0)[0]) for u in range(self.num_users_)]

        # --- Step 1: Frequent Itemset Mining ---
        self.frequent_itemsets_ = self._get_frequent_itemsets(R_bool_sets, df_bool)
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

    # The predict() method from AssociationRuleRecommender can be used directly
    def predict(self):
        """
        Generates recommendation scores based on association rules.
        The score for an item is the highest confidence of a rule
        that recommends it, based on the user's history.
        Logic based on.
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