import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class DefaultVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations by reading 'visuals.json' and dispatching
    to generic renderer functions based on 'type'.
    """
    def __init__(self, run_dir, explanations):
        super().__init__(run_dir, explanations)
        self.algorithm_name = "Default" # Set a generic name for the header

    def _render_plots(self, manifest):
        # The manifest is now passed in, so we remove all JSON loading logic

        renderer_map = {
            "line_plot": generic_renderers.render_line_plot,
            "similarity_heatmap": generic_renderers.render_similarity_heatmap,
            "histogram": generic_renderers.render_histogram,
            "factor_snapshot": generic_renderers.render_factor_snapshot,
        }

        if not manifest:
            st.info("No visualizations were generated for this run.")
            return

        for entry in manifest:
            entry_type = entry.get("type")
            renderer_func = renderer_map.get(entry_type)

            st.divider()
            if renderer_func:
                renderer_func(self.run_dir, entry, self.explanations, column=st)
            else:
                st.warning(f"No renderer found for visualization type: '{entry_type}'")