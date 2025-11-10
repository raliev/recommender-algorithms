import streamlit as st
import os
import json
from .BaseVisualizationRenderer import BaseVisualizationRenderer
from visualization import generic_renderers

class EASERenderer(BaseVisualizationRenderer):
    """
    Renders visualizations specific to the EASE algorithm
    by reading 'visuals.json' and dispatching to generic renderers.
    (Similar layout to SLIM and KNN).
    """

    def __init__(self, run_dir, explanations):
        """Initialize the renderer."""
        super().__init__(run_dir, explanations)
        self.algorithm_name = "EASE"
        self.run_timestamp = os.path.basename(run_dir)
        self.explanations.update({
            "Final Similarity Heatmap": "This heatmap shows the learned item-item similarity matrix 'B'. Unlike KNN, these values are not bounded and can be negative. A positive value at (i, j) means item j's presence contributes positively to recommending item i.",
            "Histogram of Final Similarity Values": "This histogram shows the distribution of the non-zero values in the learned 'B' matrix. EASE is a dense model, so this shows the full range of learned positive and negative item relationships.",
            "Recommendation Breakdown": "This visualization breaks down how EASE uses the learned 'B' matrix and a single user's history (R_u) to generate final scores (R_u @ B)."
        })

    def render(self):
        st.header(f"EASE Visualizations for {self.run_timestamp}")
        st.write(f"Displaying visualizations for algorithm: {self.algorithm_name}")

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

        if not manifest:
            st.info("No visualizations were generated for this run.")
            return

        # Find plots by their interpretation key
        heatmap_plot = next((e for e in manifest if e.get("interpretation_key") == "Final Similarity Heatmap"), None)
        hist_plot = next((e for e in manifest if e.get("interpretation_key") == "Histogram of Final Similarity Values"), None)
        breakdown_plot = next((e for e in manifest if e.get("type") == "recommendation_breakdown"), None)

        # --- B Matrix Properties ---
        st.subheader("Learned Model Properties (Matrix $B$)")
        col1, col2 = st.columns(2)

        if heatmap_plot:
            generic_renderers.render_similarity_heatmap(self.run_dir, heatmap_plot, self.explanations, col1)
        else:
            col1.info("Learned B matrix heatmap not found.")

        if hist_plot:
            generic_renderers.render_histogram(self.run_dir, hist_plot, self.explanations, col2)
        else:
            col2.info("Coefficient distribution plot not found.")

        st.divider()

        # --- Recommendation Example ---
        st.subheader("Recommendation Generation Example")
        if breakdown_plot:
            generic_renderers.render_recommendation_breakdown(
                self.run_dir, breakdown_plot, self.explanations, st
            )
        else:
            st.info("No recommendation breakdown plot was generated for this run.")