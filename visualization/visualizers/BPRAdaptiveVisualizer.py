import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .BPRVisualizer import BPRVisualizer

class BPRAdaptiveVisualizer(BPRVisualizer):
    """Specific visualizer for BPR (Adaptive)."""
    def __init__(self, k_factors, plot_interval=5):
        super().__init__(k_factors, plot_interval)
        self.algorithm_name = "BPR (Adaptive)"
        self.visuals_dir = os.path.join(self.visuals_base_dir, self.algorithm_name, self.run_timestamp)
        os.makedirs(self.visuals_dir, exist_ok=True)

        self.history['avg_negative_score'] = []

    def record_iteration(self, iteration_num, total_iterations, P, Q,
                         p_change, q_change, **kwargs):
        """Records BPR data and saves snapshot plots, including adaptive score."""

        # Call the base BPRVisualizer's record_iteration
        super().record_iteration(iteration_num, total_iterations, P, Q,
                                 p_change, q_change, **kwargs)

        # Record the new BPR (Adaptive) specific history key
        if 'avg_negative_score' in kwargs:
            if 'avg_negative_score' not in self.history:
                self.history['avg_negative_score'] = []
            self.history['avg_negative_score'].append(kwargs['avg_negative_score'])

    def _plot_convergence_graphs(self):
        """Plots convergence for factor change, negative score, and AUC."""

        # Call the parent BPRVisualizer's method to plot
        # factor change and AUC
        super()._plot_convergence_graphs()

        # Add the new plot for 'Avg. Negative Score'
        self._plot_convergence_line(
            history_key='avg_negative_score',
            title='Average Score of Sampled Hard Negatives',
            y_label='Average Predicted Score (x_uj)',
            filename='negative_score_convergence.png',
            interp_key='Avg. Negative Score'
        )