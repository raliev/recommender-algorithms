# visualization/visualizers/FunkSVDVisualizer.py
import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class FunkSVDVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for FunkSVD.
    Now composes plotting helpers instead of inheriting.
    """
    def __init__(self, k_factors, plot_interval=5):
        # Call AlgorithmVisualizer's init directly to set the correct name and base setup
        super().__init__("FunkSVD", plot_interval) # Use super() for cleaner inheritance
        self.k_factors = k_factors
        # FunkSVD-specific history keys (will be appended by record_iteration)
        self.history['p_change'] = []
        self.history['q_change'] = []

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def record_iteration(self, iteration_num, total_iterations, P, Q, p_change, q_change, **kwargs):
        """Records FunkSVD data and saves snapshot plots."""
        # Note: FunkSVD (as implemented) does not track a global objective value per iteration.
        # So, no 'objective' is passed to super().record_iteration.
        super().record_iteration(iteration_num) # Call base to update iteration count

        # Store factor changes
        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)

        if self._should_plot_snapshot(iteration_num, total_iterations):
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=P,
                Q=Q,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_convergence_graphs(self):
        """Plots FunkSVD convergence graphs (factor changes only)."""
        # We DON'T call super()._plot_convergence_graphs() because it plots 'objective'
        # which FunkSVD doesn't track per iteration.

        # Plot factor changes
        p_changes_to_plot = self.history['p_change'][1:] if len(self.history['p_change']) > 1 else self.history['p_change']
        q_changes_to_plot = self.history['q_change'][1:] if len(self.history['q_change']) > 1 else self.history['q_change']

        if p_changes_to_plot or q_changes_to_plot:
            manifest_entry = self.convergence_plotter.plot(
                data_dict={'P Change Norm': p_changes_to_plot, 'Q Change Norm': q_changes_to_plot},
                title='Change in Latent Factors (Frobenius Norm) over Iterations',
                y_label='Norm of Difference',
                filename='factor_change_convergence.png',
                interpretation_key='Factor Change'
            )
            self.visuals_manifest.append(manifest_entry)