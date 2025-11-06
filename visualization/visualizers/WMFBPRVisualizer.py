# visualization/visualizers/wmfbpr_visualizer.py
import os
import json
import numpy as np

from .BPRVisualizer import BPRVisualizer # Inherit from BPRVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter
from visualization.components.EmbeddingTSNEPlotter import EmbeddingTSNEPlotter
from visualization.components.VectorHistogramPlotter import VectorHistogramPlotter

class WMFBPRVisualizer(BPRVisualizer):
    """
    Specific visualizer for WMFBPR.
    Inherits from BPRVisualizer and adds a plot for the
    pre-calculated PageRank item weights.
    """
    def __init__(self, k_factors, plot_interval=5):
        # Call the parent __init__
        super().__init__(k_factors, plot_interval)
        self.algorithm_name = "WMFBPR"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True) # Ensure it exists

        # --- FIX: Re-instantiate all plotters with the new, correct directory ---
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)
        self.tsne_plotter = EmbeddingTSNEPlotter(self.visuals_dir)

        # This one was already correct
        self.histogram_plotter = VectorHistogramPlotter(self.visuals_dir)

        self.item_weights = None

    def start_run(self, params, R=None, weights=None):
        """Called at the beginning of the fit method."""
        # Now call the base start_run from AlgorithmVisualizer, not BPRVisualizer
        # to avoid BPRVisualizer's specific start_run
        super(BPRVisualizer, self).start_run(params)
        self.R = R # Store R for breakdown plot
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
        # Call the parent's end_run, which plots convergence, breakdown, etc.
        super().end_run()

        # Add the new plot
        self._plot_item_weights()

        # Re-save the manifest to include the new plot
        self._save_visuals_manifest()