import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class SVDppVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for SVD++.
    Uses components to plot convergence and factor matrix snapshots (P, Q, Y).
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("SVD++", plot_interval)
        self.k_factors = k_factors
        # SVD++ specific history keys
        self.history['objective'] = [] # Add objective key
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['y_change'] = []

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def record_iteration(self, iteration_num, total_iterations, P, Q, Y,
                         objective, # Add objective parameter
                         p_change, q_change, y_change,
                         **kwargs):
        """Records SVD++ data and saves snapshot plots."""
        # Call base to store objective AND update iteration count
        super().record_iteration(iteration_num, objective=objective)

        # Store factor changes
        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)
        self.history['y_change'].append(y_change)

        if self._should_plot_snapshot(iteration_num, total_iterations):
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=P,
                Q=Q,
                Y=Y,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_convergence_graphs(self):
        """Plots SVD++ convergence graphs (Objective and P, Q, Y factor changes)."""
        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs()

        # Plot factor changes
        p_changes_to_plot = self.history['p_change']
        q_changes_to_plot = self.history['q_change']
        y_changes_to_plot = self.history['y_change']

        # Skip first element if length > 1 to avoid large initial jump
        if len(p_changes_to_plot) > 1: p_changes_to_plot = p_changes_to_plot[1:]
        if len(q_changes_to_plot) > 1: q_changes_to_plot = q_changes_to_plot[1:]
        if len(y_changes_to_plot) > 1: y_changes_to_plot = y_changes_to_plot[1:]


        if p_changes_to_plot or q_changes_to_plot or y_changes_to_plot:
            manifest_entry = self.convergence_plotter.plot(
                data_dict={
                    'P Change Norm': p_changes_to_plot,
                    'Q Change Norm': q_changes_to_plot,
                    'Y Change Norm': y_changes_to_plot
                },
                title='Change in Latent Factors (Frobenius Norm) over Iterations',
                y_label='Norm of Difference',
                filename='factor_change_convergence.png',
                interpretation_key='Factor Change'
            )
            self.visuals_manifest.append(manifest_entry)

    def end_run(self):
        """
        Called at the end of the fit method.
        Saves params, plots convergence, saves history and manifest.
        """
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()

        self._plot_convergence_graphs() # Plot objective AND factor changes

        self._save_history()
        self._save_visuals_manifest()