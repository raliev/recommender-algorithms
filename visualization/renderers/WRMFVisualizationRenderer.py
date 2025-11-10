import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class WRMFVisualizationRenderer(BaseVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "WRMF"

    def _render_plots(self, manifest):
        self.show_convergence_plot(manifest)
        self.show_latent_factor_snapshots(manifest)
        self.show_breakdown_plot(manifest)