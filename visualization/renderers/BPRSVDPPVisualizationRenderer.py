import streamlit as st
import os
import json

from .BPRAdaptiveVisualizationRenderer import BPRAdaptiveVisualizationRenderer
from .BPRVisualizationRenderer import BPRVisualizationRenderer
from .WRMFVisualizationRenderer import WRMFVisualizationRenderer
from visualization import generic_renderers

class BPRSVDPPVisualizationRenderer(BPRAdaptiveVisualizationRenderer):
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "BPR+SVDPP"
