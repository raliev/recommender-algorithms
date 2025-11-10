import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class PureSVDVisualizationRenderer(BaseVisualizationRenderer):

    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "PureSVD"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Scree Plot": "The scree plot shows the variance explained (derived from the singular values) for each component, sorted in descending order. The 'elbow' (point of inflection) is often a good heuristic for selecting 'k'.",
            "Snapshots": "These plots visualize the distribution and relationships within the final latent factor matrices (P and Q) after decomposition."
        })

    def _render_plots(self, manifest):
        self.show_scree_plot(manifest)
        self.show_factor_matrices(manifest)