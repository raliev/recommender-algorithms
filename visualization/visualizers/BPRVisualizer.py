import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.RecommendationBreakdownPlotter import RecommendationBreakdownPlotter
from visualization.components.EmbeddingTSNEPlotter import EmbeddingTSNEPlotter # 1. IMPORT NEW PLOTTER

class BPRVisualizer(AlgorithmVisualizer):
    """Specific visualizer for BPR."""
    def __init__(self, k_factors, plot_interval=5):
        AlgorithmVisualizer.__init__(self, "BPR", plot_interval)
        self.k_factors = k_factors
        self.history['p_change'] = []
        self.history['q_change'] = []
        self.history['auc'] = [] # 2. ADD AUC TO HISTORY

        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir,
                                                  self.k_factors)
        self.breakdown_plotter = RecommendationBreakdownPlotter(self.visuals_dir)
        self.tsne_plotter = EmbeddingTSNEPlotter(self.visuals_dir) # 3. INSTANTIATE TSNE PLOTTER

        self.R = None
        self.P = None
        self.Q = None

    def start_run(self, params, R=None):
        """Called at the beginning of the fit method."""
        super().start_run(params)
        self.R = R

    def record_iteration(self, iteration_num, total_iterations, P, Q,
                         p_change, q_change, **kwargs):
        AlgorithmVisualizer.record_iteration(self, iteration_num)
        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)

        # 4. RECORD AUC IF PASSED
        if 'auc' in kwargs:
            self.history['auc'].append(kwargs['auc'])

        self.P = P
        self.Q = Q

        if self._should_plot_snapshot(iteration_num, total_iterations):
            # Plot factor snapshots
            manifest_entry_factors = self.matrix_plotter.plot_snapshot(
                P=P,
                Q=Q,
                iter_num=iteration_num,
                interpretation_key="Snapshots"
            )
            self.visuals_manifest.append(manifest_entry_factors)

            # 5. PLOT TSNE
            manifest_entry_tsne = self.tsne_plotter.plot(
                Q=Q,
                P=P,
                iter_num=iteration_num,
                title=f'Embedding t-SNE (Iter {iteration_num})',
                filename=f'tsne_iter_{iteration_num}.png',
                interpretation_key='TSNE'
            )
            if manifest_entry_tsne:
                self.visuals_manifest.append(manifest_entry_tsne)


    def _plot_convergence_graphs(self):
        # 6. MODIFIED: Plot both factor change and AUC

        # --- Plot 1: Factor Change ---
        p_changes_to_plot = self.history['p_change'][1:] if len(self.history['p_change']) > 1 else self.history['p_change']
        q_changes_to_plot = self.history['q_change'][1:] if len(self.history['q_change']) > 1 else self.history['q_change']

        if p_changes_to_plot or q_changes_to_plot:
            manifest_entry_factors = self.convergence_plotter.plot(
                data_dict={'P Change Norm': p_changes_to_plot,
                           'Q Change Norm': q_changes_to_plot},
                title='Change in Latent Factors (Frobenius Norm)',
                y_label='Norm of Difference',
                filename='factor_change_convergence.png',
                interpretation_key='Factor Change'
            )
            self.visuals_manifest.append(manifest_entry_factors)

        # --- Plot 2: AUC ---
        if 'auc' in self.history and self.history['auc']:
            auc_to_plot = self.history['auc']
            manifest_entry_auc = self.convergence_plotter.plot(
                data_dict={'Validation AUC': auc_to_plot},
                title='Validation AUC over Iterations',
                y_label='AUC',
                filename='auc_convergence.png',
                interpretation_key='AUC' # New key for renderer
            )
            self.visuals_manifest.append(manifest_entry_auc)


    def _plot_recommendation_breakdown(self):
        """
        Plots the recommendation breakdown for a single sample user.
        """
        if self.R is None or self.P is None or self.Q is None:
            print("Warning: R, P, or Q not available. "
                  "Skipping recommendation breakdown plot.")
            return

        # 1. Find a suitable sample user
        user_interaction_counts = (self.R > 0).sum(axis=1)
        sample_user_idx = np.where(
            (user_interaction_counts >= 5) & (user_interaction_counts <= 20)
        )[0]

        if len(sample_user_idx) > 0:
            sample_user_idx = sample_user_idx[0]
        elif user_interaction_counts.sum() > 0:
            sample_user_idx = np.argmax(user_interaction_counts)
        else:
            sample_user_idx = 0

            # 2. Get the necessary vectors
        user_history_vector = self.R[sample_user_idx, :]

        # Calculate full prediction scores (P_u * Q^T)
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

    def end_run(self):
        """Called at the end of the fit method."""
        self.params_saved['iterations_run'] = self.iterations_run
        self._save_params() # Update with final iteration count

        self._plot_convergence_graphs() # Plot factor changes AND AUC

        self._plot_recommendation_breakdown()

        self._save_history()
        self._save_visuals_manifest()