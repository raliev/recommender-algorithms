# visualization/visualizers/ALSImprovedVisualizer.py
import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error # To calculate RMSE objective

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter

class ALSImprovedVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for ALS (Improved) with biases.
    Plots objective (RMSE), factor change (P, Q), and matrix snapshots.
    """
    def __init__(self, k_factors, plot_interval=5):
        # --- CHANGE: Algorithm Name ---
        super().__init__("ALS (Improved)", plot_interval)
        self.k_factors = k_factors
        # History keys
        self.history['objective'] = [] # Track RMSE
        self.history['p_change'] = []
        self.history['q_change'] = []
        # Bias changes could be added here if desired

        # Instantiate components
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)

    def _calculate_objective_improved(self, R, rated_mask, P, Q, user_bias, item_bias, global_mean):
        """Calculates RMSE on the observed training ratings, including biases."""
        # Reconstruct predictions including biases
        user_bias_matrix = np.repeat(user_bias[:, np.newaxis], Q.shape[0], axis=1)
        item_bias_matrix = np.repeat(item_bias[np.newaxis, :], P.shape[0], axis=0)
        pred = global_mean + user_bias_matrix + item_bias_matrix + (P @ Q.T)

        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        if observed_actuals.size == 0:
            return 0.0 # Avoid error if rated_mask is empty
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))

    def record_iteration(self, iteration_num, total_iterations, R, rated_mask, # Pass R, mask
                         P, Q, user_bias, item_bias, global_mean, # Pass biases, mean
                         p_change, q_change, **kwargs):
        """Records ALS Improved data and saves snapshot plots."""

        # Calculate objective (RMSE) using the helper
        objective = self._calculate_objective_improved(
            R, rated_mask, P, Q, user_bias, item_bias, global_mean
        )

        # Call base to update iteration count and store objective
        super().record_iteration(iteration_num, objective=objective)

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

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs() # Plot both objective and factor changes
        self._save_history()
        self._save_visuals_manifest()