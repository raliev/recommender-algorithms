import streamlit as st
import os
import glob
import re
from abc import ABC, abstractmethod

class BaseVisualizationRenderer(ABC):
    """Abstract base class for algorithm-specific visualization renderers."""

    def __init__(self, run_dir, explanations):
        """
        Initialize the renderer.
        Args:
            run_dir (str): The path to the specific run directory containing visualization files.
            explanations (dict): A dictionary mapping plot keys to markdown explanations.
        """
        self.run_dir = run_dir
        self.explanations = explanations

    @abstractmethod
    def render(self):
        """Render the visualizations for the specific algorithm."""
        pass

    def _render_image(self, file_path, caption, explanation_key=None, column=None):
        """Helper function to render an image with optional caption and explanation."""
        container = column if column else st # Render in column or directly
        if os.path.exists(file_path):
            container.image(file_path, caption=caption)
            if explanation_key:
                with container.expander("Interpretation"):
                    st.markdown(self.explanations.get(explanation_key, 'Explanation not found.'), unsafe_allow_html=True)
        else:
            container.caption(f"{caption} not found.")
