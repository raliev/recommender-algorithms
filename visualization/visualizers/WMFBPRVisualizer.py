import os
import json
import numpy as np

from .BPRVisualizer import BPRVisualizer

class WMFBPRVisualizer(BPRVisualizer):
    """
    Specific visualizer for WMFBPR.
    Inherits from BPRVisualizer and adds a plot for the
    pre-calculated PageRank item weights.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__(k_factors, plot_interval)
        self.algorithm_name = "WMFBPR"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        self.item_weights = None

    def start_run(self, params, R=None, weights=None):
        """Called at the beginning of the fit method."""
        super(BPRVisualizer, self).start_run(params)
        self.R = R
        self.item_weights = weights

    def _plot_item_weights(self):
        """Plots the histogram of PageRank item weights."""
        if self.item_weights is not None:
            manifest_entry = self.histogram_plotter.plot(
                data_vector=self.item_weights,
                title='Distribution of Global Item Weights (w_i) from PageRank',
                filename='item_weights_histogram.png',
                interpretation_key='Item Weights',
                x_label='Normalized Item Weight (0-1)'
            )
            self.visuals_manifest.append(manifest_entry)
        else:
            print("Warning: Item weights not provided to visualizer. Skipping plot.")

    def end_run(self):
        """
        Called at the end of the fit method.
        Plots convergence, breakdown, and the new item weights.
        """
        super().end_run()

        self._plot_item_weights()

        self._save_visuals_manifest(append=True) # Append the new plot