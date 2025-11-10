import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class SVDppVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "SVD++"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Objective": "Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations. A decreasing trend indicates convergence.",
            "Factor Change": "Shows the Frobenius norm of the change in user (P), item (Q), and implicit item (Y) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P, Q, and Y) at key iterations."
        })

    def _render_plots(self, manifest):
        self.show_convergence_plot(manifest)
        self.show_latent_factor_snapshots(manifest)