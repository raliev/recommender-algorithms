import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class TopPopularVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations for TopPopular, which is a bar_plot.
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

        st.subheader("Model Internals")

        # --- Render Popularity Plot ---
        plot_entry = next((e for e in manifest if e["type"] == "bar_plot"), None)

        if plot_entry:
            # We can re-use render_line_plot as it's just a generic image renderer
            generic_renderers.render_line_plot(self.run_dir, plot_entry, self.explanations, st)
        else:
            st.info("No popularity plot found for this run.")