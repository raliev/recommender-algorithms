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
        self.history['p_change'] = []
        self.history['q_change'] = []

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir) 
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir,
                                                  self.k_factors) 
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)

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

        if self._should_plot_snapshot(iteration_num, total_iterations):
            manifest_entry = self.matrix_plotter.plot_snapshot(

                P=P, 
            Q=Q, 
            iter_num=iteration_num,
            interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry)

    def _plot_recommendation_breakdown(self):
        """
        Plots the recommendation breakdown for a single sample user.
        """
        if self.R is None or self.P is None or self.Q is None:
            print("Warning: R, P, or Q not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Find a suitable sample user
        # (Adapted from SLIMVisualizer logic )
        user_interaction_counts = (self.R > 0).sum(axis=1)
        sample_user_idx = np.where(
            (user_interaction_counts >= 5) & (user_interaction_counts <= 20)
        )[0]

        if len(sample_user_idx) > 0:
            sample_user_idx = sample_user_idx[0] # Pick first suitable user
        elif user_interaction_counts.sum() > 0:
            # Fallback to most active user
            sample_user_idx = np.argmax(user_interaction_counts)
        else:
            sample_user_idx = 0 # Fallback to user 0

        # 2. Get the necessary vectors
        user_history_vector = self.R[sample_user_idx, :]

        # Calculate full prediction scores
        all_scores = self.P @ self.Q.T
        result_vector = all_scores[sample_user_idx, :]

        num_items = self.R.shape[1]
        item_names = [f"Item {i}" for i in range(num_items)]

        # 3. Call the plotter
        manifest_entry = self.breakdown_plotter.plot(
            user_history_vector=user_history_vector,
            result_vector=result_vector,
            item_names=item_names,
            user_id=str(sample_user_idx),
            k=10, # Plot Top-10
            filename="recommendation_breakdown.png",
            interpretation_key="Recommendation Breakdown"
        )
        self.visuals_manifest.append(manifest_entry)

    def _plot_convergence_graphs(self):
        """Plots WRMF convergence graphs."""
        super()._plot_convergence_graphs() # Plot objective

        p_changes_to_plot = self.history['p_change'][1:] # ...
        q_changes_to_plot = self.history['q_change'][1:] # ...

        if p_changes_to_plot or q_changes_to_plot:
            manifest_entry = self.convergence_plotter.plot(

                data_dict={'P Change Norm': p_changes_to_plot,
                           'Q Change Norm': q_changes_to_plot}, 
            title='Change in Latent Factors (Frobenius Norm) over Iterations',
            y_label='Norm of Difference',
            filename='factor_change_convergence.png',
            interpretation_key='Factor Change'

            ) 
            self.visuals_manifest.append(manifest_entry)

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run

        self._save_params() # 

        self._plot_convergence_graphs()

        self._plot_recommendation_breakdown()

        self._save_history()
        self._save_visuals_manifest()

