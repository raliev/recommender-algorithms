import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter

class WRMFVisualizer(AlgorithmVisualizer):
    """Specific visualizer for WRMF."""
    def __init__(self, k_factors, plot_interval=5):
        super().__init__("WRMF", plot_interval)

        self.k_factors = k_factors
        self.history['objective'] = []
        self.history['p_change'] = []
        self.history['q_change'] = []

        self.R = None
        self.P = None
        self.Q = None

    def start_run(self, params, R=None):
        """Called at the beginning of the fit method."""
        super().start_run(params)
        self.R = R

    def record_iteration(self, iteration_num, total_iterations, P, Q,
                         objective, p_change, q_change):
        """Records WRMF data and saves snapshot plots."""
        super().record_iteration(iteration_num, objective=objective)
        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)

        self.P = P
        self.Q = Q

        self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_recommendation_breakdown(self):
        if self.R is None or self.P is None or self.Q is None:
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
            result_vector=result_vec
        )

    def _plot_convergence_graphs(self):
        """Plots WRMF convergence graphs."""
        super()._plot_convergence_graphs()

        # Call base to plot factor changes
        self._plot_factor_change_convergence(keys=['p_change', 'q_change'])

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run

        self._save_params() #
        self._plot_convergence_graphs()
        self._plot_recommendation_breakdown()
        self._save_history()
        self._save_visuals_manifest()

