import streamlit as st
import os
import json
from .FunkSVDVisualizationRenderer import FunkSVDVisualizationRenderer
from visualization import generic_renderers

class ALSVisualizationRenderer(FunkSVDVisualizationRenderer):
    """
    Renders visualizations specific to standard ALS.
    Inherits layout from FunkSVDVisualizationRenderer.
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "ALS"

        self.explanations.update({
            "Objective": "Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates convergence.",
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations."
        })

    def _render_plots(self, manifest):
        # Use the standard 2-column convergence helper from the base class
        self.show_convergence_plot(manifest)
        # Use the standard snapshot helper from the base class
        self.show_latent_factor_snapshots(manifest)