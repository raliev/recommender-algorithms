import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class SVDppVisualizer(AlgorithmVisualizer):
    """
     Specific visualizer for SVD++.
    Relies on base class for plotting logic.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("SVD++", plot_interval)
        self.k_factors = k_factors

        # SVD++ specific history keys
        self.history['objective'] = []
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['y_change'] = []

    def record_iteration(self, iteration_num, total_iterations, P, Q, Y,
                         objective,
                         p_change, q_change, y_change,
                         **kwargs):
        """ Records SVD++ data and saves snapshot plots."""
        # Call base to store all history AND update iteration count
        super().record_iteration(iteration_num,
                                 objective=objective,
                                 p_change=p_change,
                                 q_change=q_change,
                                 y_change=y_change)

        
        # Call generic helper from base class, passing Y
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q, Y=Y)

    def _plot_convergence_graphs(self):
        """ Plots SVD++ convergence graphs."""
        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs()
        # Call base to plot all factor changes
        self._plot_factor_change_convergence(keys=['p_change', 'q_change', 'y_change'])

    def end_run(self):
        """
        Called at the end of the fit method.
        """
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._save_history()
        self._save_visuals_manifest()