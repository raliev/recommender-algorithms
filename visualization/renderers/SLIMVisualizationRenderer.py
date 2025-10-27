# visualization/renderers/SLIMVisualizationRenderer.py
import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class SLIMVisualizationRenderer(BaseVisualizationRenderer):
    """
    Renders visualizations specific to the SLIM algorithm
    by reading 'visuals.json' and dispatching to generic renderers.
    """

    def render(self):
        manifest_path = os.path.join(self.run_dir, 'visuals.json')
        if not os.path.exists(manifest_path):
            st.warning(f"Visuals manifest 'visuals.json' not found.")
            return
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            st.error(f"Error loading 'visuals.json': {e}")
            return

        if not manifest:
            st.info("No visualizations were generated for this run.")
            return

        # Find plots by their type (as defined in the visualizer components)
        sparsity_plot = next((e for e in manifest if e.get("type") == "sparsity_plot"), None)
        hist_plot = next((e for e in manifest if e.get("type") == "histogram"), None)
        breakdown_plot = next((e for e in manifest if e.get("type") == "recommendation_breakdown"), None)

        # --- W Matrix Properties ---
        st.subheader("Learned Model Properties (Matrix $W$)")
        st.info(
            "SLIM learns a sparse item-item matrix $W$. These plots show the "
            "properties of this final learned model. The **Sparsity Plot** is the "
            "most important, showing *which* item pairs have a learned relationship."
        )
        col1, col2 = st.columns(2)

        if sparsity_plot:
            # Use a generic renderer for the image
            generic_renderers._render_image_with_interpretation(
                os.path.join(self.run_dir, sparsity_plot["file"]),
                sparsity_plot["name"],
                self.explanations.get(sparsity_plot["interpretation_key"], "Explanation not found."),
                column=col1
            )
        else:
            col1.caption("Sparsity plot not found in manifest.")

        if hist_plot:
            generic_renderers.render_histogram(self.run_dir, hist_plot, self.explanations, col2)
        else:
            col2.caption("Coefficient distribution plot not found in manifest.")

        st.divider()

        # --- Recommendation Example ---
        st.subheader("Recommendation Generation Example")
        st.info(
            "This visualization breaks down how SLIM uses the learned $W$ matrix "
            "and a single user's history ($R_u$) to generate final scores ($\tilde{R}_u$)."
        )

        if breakdown_plot:
            generic_renderers.render_recommendation_breakdown(
                self.run_dir, breakdown_plot, self.explanations, st
            )
        else:
            st.info("No recommendation breakdown plot was generated for this run.")