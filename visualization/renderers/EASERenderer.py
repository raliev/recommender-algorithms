import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class EASERenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "EASE"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Final Similarity Heatmap": "This heatmap shows the learned item-item similarity matrix 'B'. Unlike KNN, these values are not bounded and can be negative. A positive value at (i, j) means item j's presence contributes positively to recommending item i.",
            "Histogram of Final Similarity Values": "This histogram shows the distribution of the non-zero values in the learned 'B' matrix. EASE is a dense model, so this shows the full range of learned positive and negative item relationships.",
            "Recommendation Breakdown": "This visualization breaks down how EASE uses the learned 'B' matrix and a single user's history (R_u) to generate final scores (R_u @ B)."
        })

    def _render_plots(self, manifest):
        self.show_heatmap_hist_plot(manifest)
        self.show_breakdown_plot(manifest)