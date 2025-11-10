import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class NCFVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "NCFNeuMF"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Objective": "Shows the training loss (e.g., Binary Cross-Entropy) over epochs. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the embedding matrices (User and Item) at key epochs."
        })

    def _render_plots(self, manifest):
        self.show_convergence_plot(manifest)
        self.show_factor_snapshots(manifest)
        self.show_breakdown_plot(manifest)