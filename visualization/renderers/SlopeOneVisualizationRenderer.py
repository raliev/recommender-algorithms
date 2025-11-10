import os
import json
from datetime import datetime
import numpy as np
from .ItemKNNVisualizer import ItemKNNVisualizer
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter

class SlopeOneVisualizer(ItemKNNVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_name = "Slope One"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        self.similarity_plotter = SimilarityMatrixPlotter(self.visuals_dir)

    def visualize_fit_results(self, dev_matrix, freq_matrix, params=None):
        """
        Generates visualizations for Slope One:
        - Heatmap of the average deviation matrix.
        - Heatmap of the co-rated counts matrix.
        - Histogram of the deviation values.
        """
        self.start_run(params)
        self.visuals_manifest = []

        manifest_entry_dev_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=dev_matrix,
            title="Average Deviation Matrix (Sampled)",
            filename="deviation_matrix_heatmap.png",
            interpretation_key="Deviation Matrix Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_dev_heatmap)

        manifest_entry_freq_heatmap = self.similarity_plotter.plot_heatmap(
            matrix=freq_matrix,
            title="Co-rated Counts Matrix (Sampled)",
            filename="co_rated_counts_heatmap.png",
            interpretation_key="Co-rated Counts Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_freq_heatmap)

        manifest_entry_dev_hist = self.similarity_plotter.plot_histogram(
            matrix=dev_matrix,
            title="Distribution of Average Deviations (Non-Zero)",
            filename="deviation_histogram.png",
            interpretation_key="Deviation Histogram"
        )
        self.visuals_manifest.append(manifest_entry_dev_hist)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()
