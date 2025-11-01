# visualization/visualizers/BPRAdaptiveVisualizer.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .BPRVisualizer import BPRVisualizer # Inherit from BPRVisualizer
from visualization.components.ConvergencePlotter import ConvergencePlotter

class BPRAdaptiveVisualizer(BPRVisualizer):
    """Specific visualizer for BPR (Adaptive)."""
    def __init__(self, k_factors, plot_interval=5):
        # Call the parent __init__ but override the name
        super().__init__(k_factors, plot_interval)
        self.algorithm_name = "BPR (Adaptive)"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True) # Ensure it exists

        # Re-instantiate plotters with the correct directory
        self.convergence_plotter = ConvergencePlotter(self.visuals_dir)
        self.matrix_plotter = self.matrix_plotter # Can reuse parent's
        self.breakdown_plotter = self.breakdown_plotter # Can reuse parent's

        # Add new history key
        self.history['avg_negative_score'] = []

    def record_iteration(self, iteration_num, total_iterations, P, Q,
                         p_change, q_change, **kwargs):
        """Records BPR data and saves snapshot plots, including adaptive score."""
        # Call the AlgorithmVisualizer base method directly
        super(BPRVisualizer, self).record_iteration(iteration_num)

        self.history['p_change'].append(p_change)
        self.history['q_change'].append(q_change)

        # NEW: Record the average negative score
        if 'avg_negative_score' in kwargs:
            self.history['avg_negative_score'].append(kwargs['avg_negative_score'])

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

    def _plot_convergence_graphs(self):
        """Plots convergence for factor change and average negative score."""

        # --- Plot 1: Factor Change (same as original BPR) ---
        p_changes_to_plot = self.history['p_change'][1:] if len(self.history['p_change']) > 1 else self.history['p_change']
        q_changes_to_plot = self.history['q_change'][1:] if len(self.history['q_change']) > 1 else self.history['q_change']

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

        # --- Plot 2: Average Negative Score (NEW) ---
        if 'avg_negative_score' in self.history and self.history['avg_negative_score']:
            neg_scores_to_plot = self.history['avg_negative_score']
            manifest_entry_neg_score = self.convergence_plotter.plot(
                data_dict={'Avg. "Hard" Negative Score': neg_scores_to_plot},
                title='Average Score of Sampled Hard Negatives',
                y_label='Average Predicted Score (x_uj)',
                filename='negative_score_convergence.png',
                interpretation_key='Avg. Negative Score' # New key for renderer
            )
            self.visuals_manifest.append(manifest_entry_neg_score)

    # end_run, _plot_recommendation_breakdown are inherited from BPRVisualizer
    # which inherits them from AlgorithmVisualizer / WRMFVisualizer (via BPRVisualizer's parentage)
    # Let's check BPRVisualizer ... it defines its own `end_run` which calls
    # `_plot_convergence_graphs` and `_plot_recommendation_breakdown`.
