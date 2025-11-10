import streamlit as st
import os
import json

from .BPRAdaptiveVisualizationRenderer import BPRAdaptiveVisualizationRenderer
from .BPRVisualizationRenderer import BPRVisualizationRenderer
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers # Import generic renderers

class BPRSVDPPVisualizationRenderer(BPRAdaptiveVisualizationRenderer): # Inherit from WRMF
    """
    Renders visualizations specific to BPR.
    Overrides the parent render method to include AUC and t-SNE plots.
    """
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations) # Pass explanations to base
        self.algorithm_name = "BPR+SVDPP"