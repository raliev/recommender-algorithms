# visualization/renderers/FISMVisualizationRenderer.py
import streamlit as st
import os
import json
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers

class FISMVisualizationRenderer(WRMFVisualizationRenderer): # Inherit from WRMF
    """
    Renders visualizations specific to FISM.
    It inherits from WRMFVisualizationRenderer because FISM also uses latent factors
    (P and Q) and can show similar convergence and factor snapshot plots,
    plus a recommendation breakdown.
    The WRMFVisualizationRenderer's 'render' method will automatically adapt.
    """
    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        Args:
            run_dir (str): The path to the specific run directory.
            explanations (dict): A dictionary of explanations loaded from markdown.
        """
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "FISM" # Set algorithm name
        self.run_timestamp = os.path.basename(run_dir) # Get timestamp from dir

        # Add FISM-specific explanations here
        self.explanations.update({
            "Factor Change": "Shows the Frobenius norm of the change in item (P) and item (Q) latent factor matrices between iterations. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations.",
            "Recommendation Breakdown": "This visualizes the recommendation score generation (R_u * (P @ Q.T)) for a sample user."
            # Add or override more FISM-specific interpretations
        })

    # No need to override render() - the parent WRMFVisualizationRenderer.render()
    # will find "line_plot", "factor_snapshot", and "recommendation_breakdown"
    # in the manifest and render them, which is exactly what FISMVisualizer generates.