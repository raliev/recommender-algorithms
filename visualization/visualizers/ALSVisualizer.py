# visualization/visualizers/ALSVisualizer.py
import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class ALSVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for standard ALS.
    Plots objective (RMSE) convergence, factor change, and matrix snapshots.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("ALS", plot_interval)
        self.k_factors = k_factors
        # History keys for ALS
        self.history['objective'] = [] # Track RMSE
        self.history['p_change'] = []
        self.history['q_change'] = []

        # Instantiate components
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir) #
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors) #

    def record_iteration(self, iteration_num, total_iterations, P, Q, objective,
                         p_change, q_change, **kwargs):
        """Records ALS data and saves snapshot plots."""
        # Call base to update iteration count and store objective
        super().record_iteration(iteration_num, objective=objective) #

        # Store factor changes
        self.history['p_change'].append(p_change) #
        self.history['q_change'].append(q_change) #

        if self._should_plot_snapshot(iteration_num, total_iterations): #
            manifest_entry = self.matrix_plotter.plot_snapshot( #
                P=P,
                Q=Q,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry) #

    def _plot_convergence_graphs(self):
        """Plots ALS convergence graphs (Objective and Factor Changes)."""
        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs() #

        # Plot factor changes (copy from FunkSVDVisualizer)
        p_changes_to_plot = self.history['p_change'] # Use full history
        q_changes_to_plot = self.history['q_change'] # Use full history

        if p_changes_to_plot or q_changes_to_plot: #
            manifest_entry = self.convergence_plotter.plot( #
                data_dict={'P Change Norm': p_changes_to_plot, 'Q Change Norm': q_changes_to_plot},
                title='Change in Latent Factors (Frobenius Norm) over Iterations',
                y_label='Norm of Difference',
                filename='factor_change_convergence.png',
                interpretation_key='Factor Change'
            )
            self.visuals_manifest.append(manifest_entry) #

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs() # Plot both objective and factor changes
        self._save_history()
        self._save_visuals_manifest()