# visualization/visualizers/NCFVisualizer.py
import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class NCFVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for NCF / NeuMF.
    Plots loss convergence and embedding matrix snapshots.
    """
    def __init__(self, k_factors=0, plot_interval=1): # k_factors set by algorithm
        super().__init__("NCFNeuMF", plot_interval)
        self.k_factors = k_factors
        # NCF just tracks objective (loss)
        self.history['objective'] = []

        # --- Instantiate the plotting components ---
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        # k_factors must be set *after* init but before record_iteration
        # We handle this in the algorithm's fit method
        self.matrix_plotter = None

    def record_iteration(self, iteration_num, total_iterations, P, Q, objective, **kwargs):
        """Records NCF data and saves snapshot plots."""
        # Lazily initialize matrix_plotter if k_factors is now known
        if self.matrix_plotter is None and self.k_factors > 0:
            self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

        super().record_iteration(iteration_num, objective=objective)

        if self._should_plot_snapshot(iteration_num, total_iterations):
            if self.matrix_plotter and P is not None and Q is not None:
                # --- Delegate to FactorMatrixPlotter component ---
                manifest_entry = self.matrix_plotter.plot_snapshot(
                    P=P, # User Embeddings
                    Q=Q, # Item Embeddings
                    iter_num=iteration_num,
                    interpretation_key="Snapshots" # Use a common key
                )
                self.visuals_manifest.append(manifest_entry) # Add to manifest

    def _plot_convergence_graphs(self):
        """Plots NCF convergence graphs (loss only)."""
        # Call the parent method which plots 'objective'
        super()._plot_convergence_graphs()

        # end_run() is inherited from AlgorithmVisualizer and is sufficient