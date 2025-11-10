import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class ALSVisualizer(AlgorithmVisualizer):
    """
     Specific visualizer for standard ALS.
    Relies on base class for plotting logic.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("ALS", plot_interval)
        self.k_factors = k_factors

        # History keys for ALS
        self.history['objective'] = []
        self.history['p_change'] = []
        self.history['q_change'] = []

    def record_iteration(self, iteration_num, total_iterations, P, Q, objective,
                         p_change, q_change, **kwargs):
        """ Records ALS data and saves snapshot plots."""
        # Call base to update iteration count and store all history
        super().record_iteration(iteration_num,
                                 objective=objective,
                                 p_change=p_change,
                                 q_change=q_change)

        
        # Call generic helper from base class
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        """ Plots ALS convergence graphs (Objective and Factor Changes)."""
        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs() # [This was correct]
        
        # Call base to plot factor changes
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

    def end_run(self):
        """ Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._save_history()
        self._save_visuals_manifest()