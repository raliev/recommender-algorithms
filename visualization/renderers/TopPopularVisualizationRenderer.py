import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class TopPopularVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations for TopPopular, which is a bar_plot.
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "Top Popular"
        self.run_timestamp = os.path.basename(run_dir)

    def _render_plots(self, manifest):
        st.subheader("Model Internals")
        self.show_popularity_plot(manifest)