# visualization/visualizers/UserKNNVisualizer.py
import os
import json
from datetime import datetime
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter

class UserKNNVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for UserKNN.
    Now composes plotting helpers instead of containing them directly.
    """
    def __init__(self, **kwargs):
        super().__init__("UserKNN") # Set algorithm name explicitly

        self.similarity_plotter = SimilarityMatrixPlotter(self.visuals_dir)

    def visualize_fit_results(self, final_similarity_matrix, raw_similarity_matrix=None, co_rated_counts_matrix=None, params=None):
        """
        Generates visualizations for UserKNN results.
        :param final_similarity_matrix: The final, possibly adjusted, similarity matrix.
        :param raw_similarity_matrix: The initial, raw similarity matrix (optional).
        :param co_rated_counts_matrix: Matrix showing how many users were co-rated (optional).
        :param params: Parameters of the algorithm run.
        """
        self.start_run(params) # Initialize run and save params
        self.visuals_manifest = [] # Reset manifest for this visualization run

        # --- REFFACTORED: Use SimilarityMatrixPlotter component ---

        # 1. Plot Final Similarity Matrix Heatmap
        manifest_entry_final_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=final_similarity_matrix,
            title="Final (Adjusted) User-User Similarity Matrix (Sampled)",
            filename="final_user_similarity_matrix_heatmap.png",
            interpretation_key="Final (Adjusted) Similarity Heatmap" # Specific key for renderer
        )
        self.visuals_manifest.append(manifest_entry_final_heatmap)

        # 2. Plot Raw Similarity Matrix Heatmap (if provided)
        if raw_similarity_matrix is not None:
            manifest_entry_raw_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=raw_similarity_matrix,
                title="Raw User-User Similarity Matrix (Sampled)",
                filename="raw_user_similarity_matrix_heatmap.png",
                interpretation_key="Raw Similarity Heatmap" # Specific key for renderer
            )
            self.visuals_manifest.append(manifest_entry_raw_heatmap)

        # 3. Plot Co-rated Counts Matrix Heatmap (if provided)
        if co_rated_counts_matrix is not None:
            manifest_entry_co_rated_heatmap = self.similarity_plotter.plot_heatmap(
                matrix=co_rated_counts_matrix,
                title="Co-rated Users Matrix (Sampled)",
                filename="co_rated_users_matrix_heatmap.png",
                interpretation_key="Co-rated Counts Heatmap" # Specific key for renderer
            )
            self.visuals_manifest.append(manifest_entry_co_rated_heatmap)

        # 4. Plot Histogram of Final Similarity Values
        manifest_entry_hist = self.similarity_plotter.plot_histogram(
            matrix=final_similarity_matrix,
            title="Distribution of Final User-User Similarity Values (Non-Zero)",
            filename="final_user_similarity_histogram.png",
            interpretation_key="Histogram of Final Similarity Values" # Specific key for renderer
        )
        self.visuals_manifest.append(manifest_entry_hist)

        # Mark as run (no iterations for this type of visualizer)
        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history() # History might be empty, but consistent
        self._save_visuals_manifest()