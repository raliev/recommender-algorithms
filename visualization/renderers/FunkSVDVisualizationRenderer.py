# visualization/renderers/FunkSVDVisualizationRenderer.py
import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers


class FunkSVDVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations specific to FunkSVD, now reading from visuals.json
    and utilizing generic rendering components.
    """

    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        Args:
            run_dir (str): The path to the specific run directory.
            explanations (dict): A dictionary of explanations loaded from markdown.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "FunkSVD" # Set algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir
        self.explanations.update({
            # Add/update objective explanation
            "Objective": "Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates convergence.",
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations. Heatmaps show factor magnitudes, histograms show value distributions, and 2D latent space plots show user/item embeddings if k=2.",
            "Recommendation Breakdown": "This visualization breaks down how FunkSVD uses the learned latent factors (P and Q) to generate final scores for a single sample user."
        })

    def _render_plots(self, manifest):
        self.show_convergence_plot(manifest)
        self.show_error_distribution(manifest)
        self.show_factor_snapshots(manifest)
        self.show_breakdown_plot(manifest)


