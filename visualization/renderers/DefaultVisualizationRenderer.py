# visualization/renderers/DefaultVisualizationRenderer.py
import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers # Import the new generic functions

class DefaultVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations by reading 'visuals.json' and dispatching
    to generic renderer functions based on 'type'.
    """
    def render(self):
        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found in {self.run_dir}.")
            return

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            st.error(f"Error loading 'visuals.json': {e}")
            return

        # Map types to renderer functions
        renderer_map = {
            "line_plot": generic_renderers.render_line_plot,
            "similarity_heatmap": generic_renderers.render_similarity_heatmap,
            "histogram": generic_renderers.render_histogram,
            "factor_snapshot": generic_renderers.render_factor_snapshot,
            # Add more types here as needed
        }

        if not manifest:
            st.info("No visualizations were generated for this run.")
            return

        for entry in manifest:
            entry_type = entry.get("type")
            renderer_func = renderer_map.get(entry_type)

            st.divider()
            if renderer_func:
                # Call the appropriate generic renderer
                renderer_func(self.run_dir, entry, self.explanations, column=st)
            else:
                st.warning(f"No renderer found for visualization type: '{entry_type}'")