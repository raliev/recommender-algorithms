import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .AlgorithmVisualizer import AlgorithmVisualizer

class SlopeOneVisualizer(AlgorithmVisualizer):
    def __init__(self, **kwargs):
        super().__init__("Slope One")

    def visualize_fit_results(self, dev_matrix, freq_matrix, params=None):
        """
        Generates visualizations for Slope One.
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