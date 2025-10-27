# File: visualization/visualizers/PureSVDVisualizer.py
import os
import numpy as np

from .AlgorithmVisualizer import AlgorithmVisualizer
from visualization.components.FactorMatrixPlotter import FactorMatrixPlotter
from visualization.components.SingularValuesPlotter import SingularValuesPlotter

class PureSVDVisualizer(AlgorithmVisualizer):
    """
    Specific visualizer for PureSVD (and SVD).
    Plots the final factor matrices and the scree plot of singular values.
    """
    def __init__(self, k_factors, **kwargs):
        super().__init__("PureSVD")
        self.k_factors = k_factors

        # Instantiate components
        self.matrix_plotter = FactorMatrixPlotter(self.visuals_dir, self.k_factors)
        self.scree_plotter = SingularValuesPlotter(self.visuals_dir)

    def visualize_fit_results(self, P, Q, singular_values, params):
        """
        Called once by the algorithm's fit method to save all visualizations.
        """
        self.start_run(params)
        self.visuals_manifest = [] # Reset manifest

        # 1. Plot Scree Plot (Singular Values)
        manifest_entry_scree = self.scree_plotter.plot(
            singular_values=singular_values,
            k=self.k_factors,
            title="SVD Singular Values (Scree Plot)",
            filename="svd_scree_plot.png",
            interpretation_key="Scree Plot"
        )
        self.visuals_manifest.append(manifest_entry_scree)

        # 2. Plot Factor Snapshots (P and Q)
        # Use iter_num=0 to indicate this is the final state, not part of an iteration
        manifest_entry_snapshot = self.matrix_plotter.plot_snapshot(
            P=P,
            Q=Q,
            iter_num=0,
            interpretation_key="Snapshots"
        )
        self.visuals_manifest.append(manifest_entry_snapshot)

        # 3. Finalize run
        self.params_saved['iterations_run'] = 1 # Mark as 1 "step"
        self._save_params()
        self._save_history() # Saves an empty history.json, which is fine
        self._save_visuals_manifest()