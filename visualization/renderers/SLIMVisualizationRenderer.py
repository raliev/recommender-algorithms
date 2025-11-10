import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class SLIMVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "SLIM"
        self.run_timestamp = os.path.basename(run_dir)

    def _render_plots(self, manifest):
        self.show_sparcity_and_hist_plot(manifest)
        self.show_breakdown_plot(manifest)