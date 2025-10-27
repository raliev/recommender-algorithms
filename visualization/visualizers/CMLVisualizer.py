# visualization/visualizers/CMLVisualizer.py
import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class CMLVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for CML.
    Plots factor change convergence and matrix snapshots.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("CML", plot_interval) # Use super()
        self.k_factors = k_factors
        self.history['p_change'] = []
        self.history['q_change'] = []

        # --- Instantiate the plotting components ---
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def record_iteration(self, iteration_num, total_iterations, P, Q, p_change, q_change, **kwargs):
        """Records CML data and saves snapshot plots."""
        super().record_iteration(iteration_num)

        # Store factor changes
        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)

        if self._should_plot_snapshot(iteration_num, total_iterations):
            # --- Delegate to FactorMatrixPlotter component ---
            manifest_entry = self.matrix_plotter.plot_snapshot(
                P=P,
                Q=Q,
                iter_num=iteration_num,
                interpretation_key="Snapshots" # Use a common key
            )
            self.visuals_manifest.append(manifest_entry) # Add to manifest

    def _plot_convergence_graphs(self):
        """Plots CML convergence graphs (factor changes only)."""
        # Plot factor changes
        p_changes_to_plot = self.history['p_change'][1:] if len(self.history['p_change']) > 1 else self.history['p_change']
        q_changes_to_plot = self.history['q_change'][1:] if len(self.history['q_change']) > 1 else self.history['q_change']

        if p_changes_to_plot or q_changes_to_plot:
            # --- Delegate to ConvergencePlotter component ---
            manifest_entry = self.convergence_plotter.plot(
                data_dict={'P Change Norm': p_changes_to_plot, 'Q Change Norm': q_changes_to_plot},
                title='Change in Latent Factors (Frobenius Norm) over Iterations',
                y_label='Norm of Difference',
                filename='factor_change_convergence.png',
                interpretation_key='Factor Change'
            )
            self.visuals_manifest.append(manifest_entry) # Add to manifest

    def end_run(self):
        """
        Called at the end of the fit method.
        Explicitly saves params, plots, history, and the manifest.
        """
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params() # Update with final iteration count

        self._plot_convergence_graphs() # Plot CML's convergence graphs

        self._save_history()
        self._save_visuals_manifest()