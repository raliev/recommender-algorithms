import streamlit as st
import os
import json
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers

class FISMVisualizationRenderer(WRMFVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "FISM"
        self.run_timestamp = os.path.basename(run_dir)

        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in item (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations.",
            "Recommendation Breakdown": "This visualizes the recommendation score generation (R_u * (P @ Q.T)) for a sample user."
        })