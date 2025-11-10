import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class ASVDVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for AsymmetricSVD (ASVD).
    Relies on base class for plotting logic.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("ASVD", plot_interval)
        self.k_factors = k_factors

        # ASVD history keys
        self.history['objective'] = []
        self.history['q_change'] = []
        self.history['x_change'] = []
        self.history['y_change'] = []

    def record_iteration(self, iteration_num, total_iterations, Q, X, Y,
                         objective,
                         q_change, x_change, y_change,
                         **kwargs):
        """ Records ASVD data and saves snapshot plots."""
        # Call base to store all history AND update iteration count
        super().record_iteration(iteration_num,
                                 objective=objective,
                                 q_change=q_change,
                                 x_change=x_change,
                                 y_change=y_change)

        self._plot_snapshot_if_needed(iteration_num, total_iterations, P=X, Q=Q, Y=Y)

    def _plot_convergence_graphs(self):
        """ Plots ASVD convergence graphs."""

        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs() # [This was correct]

        # Call base to plot all factor changes
        self._plot_factor_change_convergence(keys=['q_change', 'x_change', 'y_change'])

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._save_history()
        self._save_visuals_manifest()