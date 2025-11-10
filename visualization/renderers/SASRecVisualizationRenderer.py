import streamlit as st
import os
import json

from visualization.renderers.BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class SASRecVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "SASRec"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Objective": "Shows the training loss (e.g., Cross-Entropy Loss) over epochs. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the Item Embedding matrix (Q) at key epochs. SASRec learns item representations rather than explicit user factors (P)."
        })

    def _render_plots(self, manifest):
        self.show_objective_plot(manifest)
        self.show_factor_snapshots(manifest)