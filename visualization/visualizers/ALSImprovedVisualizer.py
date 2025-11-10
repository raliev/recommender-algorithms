import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error # [This import is correct]
from .AlgorithmVisualizer import AlgorithmVisualizer

class ALSImprovedVisualizer(AlgorithmVisualizer):
    """
     Specific visualizer for ALS (Improved) with biases.
    Relies on base class for plotting logic.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("ALS (Improved)", plot_interval)
        self.k_factors = k_factors
        # History keys
        self.history['objective'] = []
        self.history['p_change'] = []
        self.history['q_change'] = []

    def _calculate_objective_improved(self, R, rated_mask, P, Q, user_bias, item_bias, global_mean):
        """
        Calculates RMSE on the observed training ratings, including biases.
        """
        user_bias_matrix = np.repeat(user_bias[:, np.newaxis], Q.shape[0], axis=1)
        item_bias_matrix = np.repeat(item_bias[np.newaxis, :], P.shape[0], axis=0)
        pred = global_mean + user_bias_matrix + item_bias_matrix + (P @ Q.T)
        observed_preds = pred[rated_mask]
        observed_actuals = R[rated_mask]
        if observed_actuals.size == 0:
            return 0.0
        return np.sqrt(mean_squared_error(observed_actuals, observed_preds))


    def record_iteration(self, iteration_num, total_iterations, R, rated_mask,
                         P, Q, user_bias, item_bias, global_mean,
                         p_change, q_change, **kwargs):
        """ Records ALS Improved data and saves snapshot plots."""
        objective = self._calculate_objective_improved(
            R, rated_mask, P, Q, user_bias, item_bias, global_mean
        )

        # Call base to update iteration count and store all history
        super().record_iteration(iteration_num,
                                 objective=objective,
                                 p_change=p_change,
                                 q_change=q_change)
        # Call generic helper from base class
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        """
        Plots ALS Improved convergence graphs (Objective and Factor Changes).
        """
        # Call parent to plot objective (RMSE)
        super()._plot_convergence_graphs()

        # Call base to plot factor changes
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

    def end_run(self):
        """ Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._save_history()
        self._save_visuals_manifest()