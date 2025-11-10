import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class KNNVisualizationRenderer(BaseVisualizationRenderer):

    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "KNN" # Set a generic name for the header

    def _render_plots(self, manifest):
        st.subheader("Model Internals")

        self.show_histograms_raw_final_corated_plot(manifest)
        self.show_breakdown_plot(manifest)