import os
import json
import numpy as np
from .AlgorithmVisualizer import AlgorithmVisualizer

class NCFVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for NCF / NeuMF.
    Plots loss convergence and embedding matrix snapshots.
    """
    def __init__(self, k_factors=0, plot_interval=1):
        super().__init__("NCFNeuMF", plot_interval)
        self.k_factors = k_factors
        self.history['objective'] = []

        # matrix_plotter will be lazy-loaded by the base class

        self.R = None
        self.R_predicted_final = None

    def start_run(self, params, R=None):
        """Called at the beginning of the fit method."""
        super().start_run(params)
        self.R = R

    def record_iteration(self, iteration_num, total_iterations, P, Q, objective, **kwargs):
        """Records NCF data and saves snapshot plots."""

        # Lazily set k_factors if not set during init
        if self.k_factors == 0 and P is not None:
            self.k_factors = P.shape[1]

        super().record_iteration(iteration_num, objective=objective)

        if P is not None and Q is not None:
            self._plot_snapshot_if_needed(iteration_num, total_iterations, P, Q)

    def _plot_convergence_graphs(self):
        """Plots NCF convergence graphs (loss only)."""
        super()._plot_convergence_graphs()

    def _plot_recommendation_breakdown(self, R_predicted_final):
        """
        Plots the recommendation breakdown for a single sample user using final scores.
        """
        if self.R is None or R_predicted_final is None:
            print("Warning: R or final predicted scores not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Use base helper to find the user
        user_idx, history_vec = self._find_sample_user(self.R)
        if user_idx is None:
            return

        # 2. Get the algorithm-specific score vector
        result_vec = R_predicted_final[user_idx, :]

        # 3. Use base helper to plot
        self._plot_recommendation_breakdown_generic(
            user_id=str(user_idx),
            user_history_vector=history_vec,
            result_vector=result_vec,
            interpretation_key="Recommendation Breakdown"
        )

    def end_run(self, R_predicted_final=None):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params()

        self._plot_convergence_graphs()
        self._plot_recommendation_breakdown(R_predicted_final)

        self._save_history()
        self._save_visuals_manifest()