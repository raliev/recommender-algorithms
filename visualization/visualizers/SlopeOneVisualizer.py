import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .ItemKNNVisualizer import ItemKNNVisualizer
from visualization.components.SimilarityMatrixPlotter import SimilarityMatrixPlotter

class SlopeOneVisualizer(ItemKNNVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_name = "Slope One"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)

        self.plotter = SimilarityMatrixPlotter(self.visuals_dir)

    def visualize_fit_results(self, dev_matrix, freq_matrix, params=None):
        self.start_run(params)
        self.visuals_manifest = []

        manifest_entry_1 = self.plotter.plot_heatmap(
            dev_matrix,
            "Average Deviation Matrix (Sampled)",
            "deviation_matrix_heatmap.png",
            interpretation_key="Deviation Matrix"
        )
        self.visuals_manifest.append(manifest_entry_1)

        manifest_entry_2 = self.plotter.plot_heatmap(
            freq_matrix,
            "Co-rated Counts Matrix (Sampled)",
            "co_rated_counts_heatmap.png",
            interpretation_key="Co-rated Counts Heatmap"
        )
        self.visuals_manifest.append(manifest_entry_2)

        manifest_entry_3 = self.plotter.plot_histogram(
            dev_matrix,
            "Distribution of Average Deviations (Non-Zero)",
            "deviation_histogram.png",
            interpretation_key="Deviation Histogram"
        )
        self.visuals_manifest.append(manifest_entry_3)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()