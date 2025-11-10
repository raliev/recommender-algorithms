import os
import numpy as np
from .AlgorithmVisualizer import AlgorithmVisualizer

class SASRecVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for SASRec.
    Plots loss convergence and item embedding snapshots.
    """
    def __init__(self, k_factors, plot_interval=1):
        super().__init__("SASRec", plot_interval)
        self.k_factors = k_factors
        self.history['objective'] = []

    def record_iteration(self, iteration_num, total_iterations, objective, Q, P=None, **kwargs):
        """Records SASRec data (loss) and saves item embedding snapshot plots."""
        super().record_iteration(iteration_num, objective=objective)

        # Plot snapshot of item embeddings (Q), passing P as None
        self._plot_snapshot_if_needed(iteration_num, total_iterations, P=P, Q=Q)

    def _plot_convergence_graphs(self):
        """Plots SASRec convergence graphs (loss only)."""
        super()._plot_convergence_graphs()