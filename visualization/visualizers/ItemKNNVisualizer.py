# visualization/visualizers/ItemKNNVisualizer.py
import os
import json
from datetime import datetime
import numpy as np
# No longer importing matplotlib or seaborn directly here for plotting helpers

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter


class ItemKNNVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for ItemKNN.
    Now composes plotting helpers instead of containing them directly.
    """
    def __init__(self, **kwargs):
        super().__init__("ItemKNN") # Set algorithm name explicitly
        # ItemKNN doesn't usually have plot_interval or iterations, but AlgorithmVisualizer expects it
        # We can pass 0 or a large number for plot_interval if we don't need iteration-based plots.
        # For simplicity, let's keep it as super().__init__("ItemKNN") and rely on visualize_fit_results.

        self.similarity_plotter = SimilarityMatrixPlotter(self.visuals_dir)
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)

    def _plot_recommendation_breakdown(self, R, final_similarity_matrix):
        """
        Plots the recommendation breakdown for a single sample user
        using a simplified R * S calculation.
        """
        if R is None or final_similarity_matrix is None:
            print("Warning: R or similarity matrix not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Find a suitable sample user (logic from WRMF/BPR)
        user_interaction_counts = (R > 0).sum(axis=1)
        sample_user_idx = np.where(
            (user_interaction_counts >= 5) & (user_interaction_counts <= 20)
        )[0]

        if len(sample_user_idx) > 0:
            sample_user_idx = sample_user_idx[0]
        elif user_interaction_counts.sum() > 0:
            sample_user_idx = np.argmax(user_interaction_counts)
        else:
            sample_user_idx = 0

        # 2. Get the necessary vectors
        user_history_vector = R[sample_user_idx, :]

        # 3. Calculate simplified prediction scores (R_u * S)
        # This approximates the contribution of all items
        result_vector = user_history_vector @ final_similarity_matrix

        num_items = R.shape[1]
        item_names = [f"Item {i}" for i in range(num_items)]

        # 4. Call the plotter
        manifest_entry = self.breakdown_plotter.plot(
            user_history_vector=user_history_vector,
            result_vector=result_vector,
            item_names=item_names,
            user_id=str(sample_user_idx),
            k=10, # Plot Top-10
            filename="recommendation_breakdown.png",
            interpretation_key="Recommendation Breakdown"
        )
        self.visuals_manifest.append(manifest_entry)

    def visualize_fit_results(self, R, final_similarity_matrix=None,
                              raw_similarity_matrix=None,
                              co_rated_counts_matrix=None, params=None):
        """
        Generates visualizations for ItemKNN results.
        :param R: The training user-item matrix
        :param final_similarity_matrix: The final, possibly adjusted, similarity matrix.
        :param raw_similarity_matrix: The initial, raw similarity matrix (optional).
        :param co_rated_counts_matrix: Matrix showing how many items were co-rated (optional).
        :param params: Parameters of the algorithm run.
        """
        self.start_run(params) # Initialize run and save params
        self.visuals_manifest = [] # Reset manifest for this visualization run

        # --- (Similarity and Histogram plotting logic remains the same) ---
        # 1. Plot Final Similarity Matrix Heatmap
        manifest_entry_final_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=final_similarity_matrix,
            title="Final (Adjusted) Item-Item Similarity Matrix (Sampled)",
            filename="final_similarity_matrix_heatmap.png",
            interpretation_key="Final (Adjusted) Similarity Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_final_heatmap)

        # 2. Plot Raw Similarity Matrix Heatmap (if provided)
        if raw_similarity_matrix is not None:
            # ... (append to manifest)
            manifest_entry_raw_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=raw_similarity_matrix,
                title="Raw Item-Item Similarity Matrix (Sampled)",
                filename="raw_similarity_matrix_heatmap.png",
                interpretation_key="Raw Similarity Heatmap"
            )
            self.visuals_manifest.append(manifest_entry_raw_heatmap)

        # 3. Plot Co-rated Counts Matrix Heatmap (if provided)
        if co_rated_counts_matrix is not None:
            # ... (append to manifest)
            manifest_entry_co_rated_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=co_rated_counts_matrix,
                title="Co-rated Counts Matrix (Sampled)",
                filename="co_rated_counts_heatmap.png",
                interpretation_key="Co-rated Counts Heatmap"
            )
            self.visuals_manifest.append(manifest_entry_co_rated_heatmap)

        # 4. Plot Histogram of Final Similarity Values
        manifest_entry_hist = self.similarity_plotter.plot_histogram(
            matrix=final_similarity_matrix,
            title="Distribution of Final Item-Item Similarity Values (Non-Zero)",
            filename="final_similarity_histogram.png",
            interpretation_key="Histogram of Final Similarity Values"
        )
        self.visuals_manifest.append(manifest_entry_hist)

        self._plot_recommendation_breakdown(R, final_similarity_matrix)

        # Mark as run (no iterations for this type of visualizer)
        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history() # History might be empty, but consistent
        self._save_visuals_manifest()