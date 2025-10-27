# visualization/visualizers/SVDppVisualizer.py
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
        # Call AlgorithmVisualizer's init directly
        super().__init__("SVD++", plot_interval)
        self.k_factors = k_factors
        # SVD++ specific history keys
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['y_change'] = []

        # --- Instantiate components ---
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def record_iteration(self, iteration_num, total_iterations, P, Q, Y,
                         p_change, q_change, y_change,
                         **kwargs):
        """Records SVD++ data and saves snapshot plots."""
        # No 'objective' tracked, just iteration number
        super().record_iteration(iteration_num)

        # Store factor changes
        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)
        self.history['y_change'].append(y_change)

        if self._should_plot_snapshot(iteration_num, total_iterations):
            # --- Delegate to FactorMatrixPlotter, passing Y ---
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=P,
                Q=Q,
                Y=Y,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_convergence_graphs(self):
        """Plots SVD++ convergence graphs (P, Q, and Y factor changes)."""
        p_changes_to_plot = self.history['p_change']
        q_changes_to_plot = self.history['q_change']
        y_changes_to_plot = self.history['y_change']

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
            self.visuals_manifest.append(manifest_entry) #

    # --- Add explicit end_run for clarity ---
    def end_run(self):
        """
        Called at the end of the fit method.
        Saves params, plots convergence, saves history and manifest.
        """
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()

        self._plot_convergence_graphs() # Plot P, Q, Y factor changes

        self._save_history()
        self._save_visuals_manifest()