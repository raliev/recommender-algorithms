import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .AlgorithmVisualizer import AlgorithmVisualizer

class FISMVisualizer(AlgorithmVisualizer):
    """Specific visualizer for FISM."""
    def __init__(self, k_factors, plot_interval=5):
        AlgorithmVisualizer.__init__(self, "FISM", plot_interval)
        self.k_factors = k_factors
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['objective'] = [] # Added for objective tracking

        self.R = None # To store training matrix
        self.P = None # To store final item-factors P
        self.Q = None # To store final item-factors Q

    def start_run(self, params, R=None):
        """Called at the beginning of the fit method."""
        super().start_run(params)
        self.R = R

    def record_iteration(self, iteration_num, total_iterations, P, Q,
                         p_change, q_change, **kwargs):

        # Call base to store history
        super().record_iteration(iteration_num,
                                 p_change=p_change,
                                 q_change=q_change,
                                 objective=kwargs.get('objective'))

        self.P = P
        self.Q = Q

        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        """Plots FISM convergence (objective and factor changes)."""

        # Call base to plot objective
        super()._plot_convergence_graphs()

        # Call base to plot factor changes
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

    def _plot_recommendation_breakdown(self):
        """
        Plots the recommendation breakdown for a single sample user.
        """
        if self.R is None or self.P is None or self.Q is None:
            print("Warning: R, P, or Q not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Use base helper to find the user
        user_idx, history_vec = self._find_sample_user(self.R)
        if user_idx is None:
            return

        # 2. Calculate the algorithm-specific score vector
        # FISM's score is R_u @ (P @ Q.T)
        similarity_matrix = self.P @ self.Q.T
        result_vec = history_vec @ similarity_matrix

        # 3. Use base helper to plot
        self._plot_recommendation_breakdown_generic(
            user_id=str(user_idx),
            user_history_vector=history_vec,
            result_vector=result_vec,
            interpretation_key="Recommendation Breakdown"
        )

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._plot_recommendation_breakdown()
        self._save_history()
        self._save_visuals_manifest()