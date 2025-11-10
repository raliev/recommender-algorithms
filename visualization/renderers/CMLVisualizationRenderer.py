import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class CMLVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "CML"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations."
        })

    def _render_plots(self, manifest):
        self.show_factor_1column(manifest)
        self.show_factor_snapshots(manifest)