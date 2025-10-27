# visualization/renderers/ALSImprovedVisualizationRenderer.py
import streamlit as st
import os
from .ALSVisualizationRenderer import ALSVisualizationRenderer # Inherit from standard ALS

class ALSImprovedVisualizationRenderer(ALSVisualizationRenderer): # Inherit
    """
    Renders visualizations specific to ALS (Improved).
    Inherits layout directly from ALSVisualizationRenderer as the plots are the same.
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations) # Call parent __init__
        self.algorithm_name = "ALS (Improved)" # Set correct name
        # Update explanations if needed, e.g., mention biases aren't plotted
        self.explanations.update({ #
            "Objective": "Shows the Root Mean Squared Error (RMSE) calculated on the *observed ratings* in the training set over iterations (including bias terms). A decreasing trend indicates convergence.",
            "Factor Change": "Shows the Frobenius norm of the change in user (P) and item (Q) latent factor matrices between iterations. Bias changes are not shown here. A decreasing trend indicates convergence.",
            "Snapshots": "These plots visualize the distribution and relationships within the latent factor matrices (P and Q) at key iterations."
        })
