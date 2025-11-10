import streamlit as st
from .PureSVDVisualizationRenderer import PureSVDVisualizationRenderer
class SVDVisualizationRenderer(PureSVDVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "SVD"