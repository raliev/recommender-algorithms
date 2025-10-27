# visualization/visualizers/SlopeOneVisualizer.py
import os
import json
from datetime import datetime
import numpy as np # Needed for potential matrix operations
from .ItemKNNVisualizer import ItemKNNVisualizer # Still inherits for base setup
# Import the new plotting component
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter

class SlopeOneVisualizer(ItemKNNVisualizer): # Inheriting from ItemKNNVisualizer for base visualizer setup
    def __init__(self, **kwargs):
        # ItemKNNVisualizer's __init__ will set up self.algorithm_name, self.visuals_dir, etc.
        super().__init__(**kwargs)
        self.algorithm_name = "Slope One" # Ensure algorithm name is correct for output paths
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True) # Ensure it exists for this specific algo

        self.similarity_plotter = SimilarityMatrixPlotter(self.visuals_dir)

    def visualize_fit_results(self, dev_matrix, freq_matrix, params=None):
        """
        Generates visualizations for Slope One:
        - Heatmap of the average deviation matrix.
        - Heatmap of the co-rated counts matrix.
        - Histogram of the deviation values.
        """
        self.start_run(params) # Initialize run and save params
        self.visuals_manifest = [] # Reset manifest for this visualization run

        # --- REFFACTORED: Use SimilarityMatrixPlotter component ---

        # 1. Plot Average Deviation Matrix Heatmap
        manifest_entry_dev_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=dev_matrix,
            title="Average Deviation Matrix (Sampled)",
            filename="deviation_matrix_heatmap.png", # Logical filename
            interpretation_key="Deviation Matrix Heatmap" # Specific key for interpretation
        )
        self.visuals_manifest.append(manifest_entry_dev_heatmap)

        # 2. Plot Co-rated Counts Matrix Heatmap
        manifest_entry_freq_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=freq_matrix,
            title="Co-rated Counts Matrix (Sampled)",
            filename="co_rated_counts_heatmap.png", # Logical filename
            interpretation_key="Co-rated Counts Heatmap" # Specific key for interpretation
        )
        self.visuals_manifest.append(manifest_entry_freq_heatmap)

        # 3. Plot Histogram of Deviation Values
        manifest_entry_dev_hist = self.similarity_plotter.plot_histogram(
            matrix=dev_matrix,
            title="Distribution of Average Deviations (Non-Zero)",
            filename="deviation_histogram.png", # Logical filename
            interpretation_key="Deviation Histogram" # Specific key for interpretation
        )
        self.visuals_manifest.append(manifest_entry_dev_hist)

        # We don't have iterations for Slope One fit, so set to 1
        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history() # History might be empty for Slope One, but it's consistent
        self._save_visuals_manifest() # Save the collected manifest
