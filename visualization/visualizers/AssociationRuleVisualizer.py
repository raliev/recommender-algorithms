import os
import json
import pandas as pd
from .AlgorithmVisualizer import AlgorithmVisualizer

class AssociationRuleVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for Association Rule models (Apriori, FP-Growth, Eclat).
    Saves the frequent itemsets and association rules DataFrames as JSON.
    """

    def __init__(self, algorithm_name, **kwargs):
        # Pass the specific algorithm name (e.g., "Apriori")
        super().__init__(algorithm_name)

    def _save_dataframe_to_json(self, df, filename):
        """Helper to save DataFrame as JSON."""
        # Convert frozensets to lists for JSON serialization
        df_copy = df.copy()
        for col in df_copy.columns:
            # Check if all items in the column are frozensets
            if not df_copy[col].empty and all(isinstance(item, frozenset) for item in df_copy[col]):
                df_copy[col] = df_copy[col].apply(list)

        path = os.path.join(self.visuals_dir, filename)
        df_copy.to_json(path, orient='records', indent=4)
        return filename

    def visualize_fit_results(self, frequent_itemsets, rules, params):
        """
        Called once by the algorithm's fit method.
        """
        self.start_run(params) # Creates dir, saves params 
        self.visuals_manifest = []

        # 1. Save Frequent Itemsets
        if not frequent_itemsets.empty:
            itemsets_file = self._save_dataframe_to_json(
                frequent_itemsets.nlargest(50, 'support'),
                "frequent_itemsets_top50.json"
            )
            self.visuals_manifest.append({
                "name": "Top 50 Frequent Itemsets (by Support)",
                "type": "dataframe",
                "file": itemsets_file,
                "interpretation_key": "Frequent Itemsets"
            })

        # 2. Save Association Rules
        if not rules.empty:
            rules_file = self._save_dataframe_to_json(
                rules.nlargest(100, 'confidence'),
                "association_rules_top100.json"
            )
            self.visuals_manifest.append({
                "name": "Top 100 Association Rules (by Confidence)",
                "type": "dataframe",
                "file": rules_file,
                "interpretation_key": "Association Rules"
            })

        # Finalize run
        self.params_saved['iterations_run'] = 1 # Mark as 1 "step"
        self._save_params() # 
        self._save_history() # Saves an empty history.json 
        self._save_visuals_manifest() # 