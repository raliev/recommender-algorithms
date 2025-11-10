import os
import json
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class CMLVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for CML.
    Relies on base class for plotting logic.
    """
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("CML", plot_interval)
        self.k_factors = k_factors
        self.history['p_change'] = []
        self.history['q_change'] = []

        self.R = None
        self.P = None
        self.Q = None

    def start_run(self, params, R=None):
        """[NO CHANGE NEEDED] Called at the beginning of the fit method."""
        super().start_run(params)
        self.R = R

    def record_iteration(self, iteration_num, total_iterations, P, Q, p_change, q_change, **kwargs):
        """[MODIFIED] Records CML data and saves snapshot plots."""

        # Call base to store history
        super().record_iteration(iteration_num,
                                 p_change=p_change,
                                 q_change=q_change)

        # Store final P and Q for breakdown plot
        self.P = P
        self.Q = Q

        # Call generic helper from base class
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        """Plots CML convergence graphs (factor changes only)."""
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

    def _plot_recommendation_breakdown(self):
        """
        [MODIFIED] Plots the recommendation breakdown using negative distance scores.
        """
        if self.R is None or self.P is None or self.Q is None:
            print("Warning: R, P, or Q not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Use base helper to find the user
        user_idx, history_vec = self._find_sample_user(self.R)
        if user_idx is None:
            return

        # 2. Calculate the algorithm-specific score vector (negative distance)
        result_vec = -np.linalg.norm(self.P[user_idx, np.newaxis, :] - self.Q, axis=1)

        # 3. Use base helper to plot
        self._plot_recommendation_breakdown_generic(
            user_id=str(user_idx),
            user_history_vector=history_vec,
            result_vector=result_vec,
            interpretation_key="Recommendation Breakdown"
        )

    def end_run(self):
        """ Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._plot_recommendation_breakdown()
        self._save_history()
        self._save_visuals_manifest()