# File: visualization/renderers/SVDVisualizationRenderer.py
import streamlit as st
from .PureSVDVisualizationRenderer import PureSVDVisualizationRenderer

class SVDVisualizationRenderer(PureSVDVisualizationRenderer): # Inherit from PureSVD
    """
    Renders visualizations specific to SVD.
    Inherits the render() logic from PureSVDVisualizationRenderer.
    """
    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations) # Call parent __init__
        self.algorithm_name = "SVD" # Set correct name
        # Explanations are inherited from parent and are suitable for SVD.

    # The render() method is inherited from PureSVDVisualizationRenderer
    # and will automatically use self.algorithm_name = "SVD" in the header.