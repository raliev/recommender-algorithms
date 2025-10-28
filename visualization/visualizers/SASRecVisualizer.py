# visualization/visualizers/SASRecVisualizer.py
import os
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter # Reuse for embedding heatmap

class SASRecVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for SASRec.
    Plots loss convergence and item embedding snapshots.
    """
    def __init__(self, k_factors, plot_interval=1): # plot_interval often 1 for epochs
        super().__init__("SASRec", plot_interval)
        self.k_factors = k_factors
        # SASRec primarily tracks objective (loss)
        self.history['objective'] = []

        # Instantiate components
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        # Use FactorMatrixPlotter to visualize the item embedding matrix Q
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def record_iteration(self, iteration_num, total_iterations, objective, Q, P=None, **kwargs):
        """Records SASRec data (loss) and saves item embedding snapshot plots."""
        # Call base to update iteration count and store objective
        super().record_iteration(iteration_num, objective=objective)

        # Plot snapshot of item embeddings (Q)
        if self._should_plot_snapshot(iteration_num, total_iterations):
            # Pass P=None as SASRec doesn't have a distinct user matrix in the same way
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=P, # Can be None or zeros
                Q=Q, # Item embeddings
                iter_num=iteration_num,
                interpretation_key="Snapshots" # Generic key
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_convergence_graphs(self):
        """Plots SASRec convergence graphs (loss only)."""
        # Call the parent method which plots 'objective'
        super()._plot_convergence_graphs()