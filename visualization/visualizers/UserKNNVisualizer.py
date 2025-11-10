import os
import json
from datetime import datetime
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class UserKNNVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for UserKNN.
    """
    def __init__(self, **kwargs):
        super().__init__("UserKNN")

    def visualize_fit_results(self, final_similarity_matrix, raw_similarity_matrix=None, co_rated_counts_matrix=None, params=None):
        """
        Generates visualizations for UserKNN results.
        """
        self.start_run(params)
        self.visuals_manifest = []

        manifest_entry_final_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=final_similarity_matrix,
            title="Final (Adjusted) User-User Similarity Matrix (Sampled)",
            filename="final_user_similarity_matrix_heatmap.png",
            interpretation_key="Final (Adjusted) Similarity Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_final_heatmap)

        if raw_similarity_matrix is not None:
            manifest_entry_raw_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=raw_similarity_matrix,
                title="Raw User-User Similarity Matrix (Sampled)",
                filename="raw_user_similarity_matrix_heatmap.png",
                interpretation_key="Raw Similarity Heatmap"
            )
            self.visuals_manifest.append(manifest_entry_raw_heatmap)

        if co_rated_counts_matrix is not None:
            manifest_entry_co_rated_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=co_rated_counts_matrix,
                title="Co-rated Users Matrix (Sampled)",
                filename="co_rated_users_matrix_heatmap.png",
                interpretation_key="Co-rated Counts Heatmap"
            )
            self.visuals_manifest.append(manifest_entry_co_rated_heatmap)

        manifest_entry_hist = self.similarity_plotter.plot_histogram(
            matrix=final_similarity_matrix,
            title="Distribution of Final User-User Similarity Values (Non-Zero)",
            filename="final_user_similarity_histogram.png",
            interpretation_key="Histogram of Final Similarity Values"
        )
        self.visuals_manifest.append(manifest_entry_hist)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()