import os
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer

class PureSVDVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for PureSVD (and SVD).
    Plots the final factor matrices and the scree plot of singular values.
    """
    def __init__(self, k_factors, **kwargs):
        super().__init__("PureSVD")
        self.k_factors = k_factors

    def visualize_fit_results(self, P, Q, singular_values, params):
        """
        Called once by the algorithm's fit method to save all visualizations.
        """
        self.start_run(params)
        self.visuals_manifest = []

        manifest_entry_scree = self.scree_plotter.plot(
            singular_values=singular_values,
            k=self.k_factors,
            title="SVD Singular Values (Scree Plot)",
            filename="svd_scree_plot.png",
            interpretation_key="Scree Plot"
        )
        self.visuals_manifest.append(manifest_entry_scree)

        manifest_entry_snapshot = self.matrix_plotter.plot_snapshot(
            P=P,
            Q=Q,
            iter_num=0,
            interpretation_key="Snapshots"
        )
        self.visuals_manifest.append(manifest_entry_snapshot)

        self.params_saved['iterations_run'] = 1
        self._save_params()
        self._save_history()
        self._save_visuals_manifest()