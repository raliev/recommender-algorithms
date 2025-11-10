import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class ASVDVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for AsymmetricSVD (ASVD).
    Plots convergence and factor matrix snapshots (Q, X, Y).
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("ASVD", plot_interval)
        self.k_factors = k_factors
        # ASVD history keys
        self.history['objective'] = []
        self.history['q_change'] = []
        self.history['x_change'] = []
        self.history['y_change'] = []

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def record_iteration(self, iteration_num, total_iterations, Q, X, Y,
                         objective,
                         q_change, x_change, y_change,
                         **kwargs):
        """Records ASVD data and saves snapshot plots."""
        super().record_iteration(iteration_num, objective=objective)

        # Store factor changes
        self.history['q_change'].append(q_change)
        self.history['x_change'].append(x_change)
        self.history['y_change'].append(y_change)

        if self._should_plot_snapshot(iteration_num, total_iterations):
            # We map ASVD's (X, Q, Y) to the plotter's (P, Q, Y)
            # P = X (Explicit Item Factors)
            # Q = Q (Item Factors)
            # Y = Y (Implicit Item Factors)
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=X, # Pass X as P
                Q=Q,
                Y=Y,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_convergence_graphs(self):
        """Plots ASVD convergence graphs (Objective and Q, X, Y factor changes)."""
        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs()

        # Plot factor changes
        q_changes_to_plot = self.history['q_change']
        x_changes_to_plot = self.history['x_change']
        y_changes_to_plot = self.history['y_change']

        if len(q_changes_to_plot) > 1: q_changes_to_plot = q_changes_to_plot[1:]
        if len(x_changes_to_plot) > 1: x_changes_to_plot = x_changes_to_plot[1:]
        if len(y_changes_to_plot) > 1: y_changes_to_plot = y_changes_to_plot[1:]

        if q_changes_to_plot or x_changes_to_plot or y_changes_to_plot:
            manifest_entry = self.convergence_plotter.plot(
                data_dict={
                    'Q Change Norm': q_changes_to_plot,
                    'X Change Norm': x_changes_to_plot,
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
        """
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._save_history()
        self._save_visuals_manifest()