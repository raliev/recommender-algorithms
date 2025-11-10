import os
import json
from datetime import datetime
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class ItemKNNVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for ItemKNN.
    """
    def __init__(self, **kwargs):
        super().__init__("ItemKNN")

    def _plot_recommendation_breakdown(self, R, final_similarity_matrix):
        """
        Plots the recommendation breakdown for a single sample user
        using a simplified R * S calculation.
        """
        if R is None or final_similarity_matrix is None:
            print("Warning: R or similarity matrix not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Use base helper to find the user
        user_idx, history_vec = self._find_sample_user(R)
        if user_idx is None:
            return

        # 2. Calculate the algorithm-specific score vector
        result_vec = history_vec @ final_similarity_matrix

        # 3. Use base helper to plot
        self._plot_recommendation_breakdown_generic(
            user_id=str(user_idx),
            user_history_vector=history_vec,
            result_vector=result_vec,
            interpretation_key="Recommendation Breakdown"
        )

    def visualize_fit_results(self, R, final_similarity_matrix=None,
                              raw_similarity_matrix=None,
                              co_rated_counts_matrix=None, params=None):
        """
        Generates visualizations for ItemKNN results.
        """
        self.start_run(params)
        self.visuals_manifest = []

        manifest_entry_final_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=final_similarity_matrix,
            title="Final (Adjusted) Item-Item Similarity Matrix (Sampled)",
            filename="final_similarity_matrix_heatmap.png",
            interpretation_key="Final (Adjusted) Similarity Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_final_heatmap)

        if raw_similarity_matrix is not None:
            manifest_entry_raw_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=raw_similarity_matrix,
                title="Raw Item-Item Similarity Matrix (Sampled)",
                filename="raw_similarity_matrix_heatmap.png",
                interpretation_key="Raw Similarity Heatmap"
            )
            self.visuals_manifest.append(manifest_entry_raw_heatmap)

        if co_rated_counts_matrix is not None:
            manifest_entry_co_rated_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=co_rated_counts_matrix,
                title="Co-rated Counts Matrix (Sampled)",
                filename="co_rated_counts_heatmap.png",
                interpretation_key="Co-rated Counts Heatmap"
            )
            self.visuals_manifest.append(manifest_entry_co_rated_heatmap)

        manifest_entry_hist = self.similarity_plotter.plot_histogram(
            matrix=final_similarity_matrix,
            title="Distribution of Final Item-Item Similarity Values (Non-Zero)",
            filename="final_similarity_histogram.png",
            interpretation_key="Histogram of Final Similarity Values"
        )
        self.visuals_manifest.append(manifest_entry_hist)

        self._plot_recommendation_breakdown(R, final_similarity_matrix)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()