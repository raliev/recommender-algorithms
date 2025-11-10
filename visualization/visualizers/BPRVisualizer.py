import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .AlgorithmVisualizer import AlgorithmVisualizer

class BPRVisualizer(AlgorithmVisualizer):
    def __init__(self, k_factors, plot_interval=5):
        AlgorithmVisualizer.__init__(self, "BPR", plot_interval)
        self.k_factors = k_factors
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['auc'] = [] # [This was correct]

        self.R = None
        self.P = None
        self.Q = None

    def start_run(self, params, R=None):
        super().start_run(params)
        self.R = R

    def record_iteration(self, iteration_num, total_iterations, P, Q, p_change, q_change, **kwargs):
        # Call base to store history
        super().record_iteration(iteration_num, # Corrected: removed 'self'
                                 p_change=p_change,
                                 q_change=q_change,
                                 auc=kwargs.get('auc'))

        # Store matrices for final breakdown plot
        self.P, self.Q = P, Q

        # Call generic plotters
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)
        self._plot_tsne_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        # Plot p_change and q_change using the new base helper
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

        # Plot auc using the new base helper
        self._plot_convergence_line(
            history_key='auc',
            title='Validation AUC over Iterations',
            y_label='AUC',
            filename='auc_convergence.png',
            interp_key='AUC'
        )

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
        all_scores = self.P @ self.Q.T
        result_vec = all_scores[user_idx, :]

        # 3. Use base helper to plot
        self._plot_recommendation_breakdown_generic(
            user_id=str(user_idx),
            user_history_vector=history_vec,
            result_vector=result_vec,
            interpretation_key="Recommendation Breakdown"
        )

    def end_run(self):
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()
        self._plot_convergence_graphs()
        self._plot_recommendation_breakdown()
        self._save_history()
        self._save_visuals_manifest()